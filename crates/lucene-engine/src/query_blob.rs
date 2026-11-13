//! The query blob a request carries, decoded into a query.
//!
//! A request is one blob: an optional `min_score` and per-field similarity
//! table in front of a term, a boolean in the occur-tagged clause format, or
//! a query tree of recursive nodes ([`decode_node`]). Everything is
//! little-endian and bounds-checked by [`Cursor`]; a malformed blob is
//! [`Status::InvalidArgument`] with a message, never a panic.

use lucene_search::extended_query::{
    MultiTermQuery, MultiTermSource, RewriteMethod, TermRangeQuery,
};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::similarities::PerFieldSimilarity;

use crate::alloc::try_with_capacity;
use crate::clauses::{check_clause_count, read_boolean_clauses};
use crate::status::{set_last_error, Status};

/// Blob tag for a single `TermQuery`.
pub const QUERY_TERM: u8 = 0;
/// Blob tag for a `BooleanQuery` in the occur-tagged clause format.
pub const QUERY_BOOLEAN: u8 = 1;
/// Blob tag for a query tree: one recursive node, see [`decode_node`].
pub const QUERY_TREE: u8 = 2;
/// Blob tag for OpenSearch's `min_score` in front of a query: the minimum
/// (`f32`), then the query's own blob ([`decode_request`]).
pub const QUERY_MIN_SCORE: u8 = 3;

/// Query-tree node kinds ([`QUERY_TREE`]).
pub const NODE_TERM: u8 = 0;
pub const NODE_BOOLEAN: u8 = 1;
pub const NODE_CONSTANT_SCORE: u8 = 2;
pub const NODE_BOOST: u8 = 3;
pub const NODE_DISMAX: u8 = 4;
pub const NODE_MATCH_ALL: u8 = 5;
pub const NODE_MATCH_NONE: u8 = 6;
pub const NODE_PHRASE: u8 = 7;
pub const NODE_TERM_SET: u8 = 8;
pub const NODE_PREFIX: u8 = 9;
pub const NODE_WILDCARD: u8 = 10;
pub const NODE_POINT_RANGE: u8 = 11;
pub const NODE_REGEXP: u8 = 12;
pub const NODE_EXISTS: u8 = 13;
pub const NODE_TERM_STATS: u8 = 14;
pub const NODE_POINTS_BOX: u8 = 15;
pub const NODE_GEO_DISTANCE: u8 = 16;
pub const NODE_GEO_POINT: u8 = 17;
pub const NODE_GEO_SHAPE: u8 = 18;
pub const NODE_GEO_SHAPE_BOX: u8 = 19;
/// `TermRangeQuery` under its constant-score rewrite; follows
/// [`crate::nodes::NODE_LAST`].
pub const NODE_TERM_RANGE: u8 = 25;
const _: () = assert!(NODE_TERM_RANGE == crate::nodes::NODE_LAST + 1);
/// `PointInSetQuery` (`terms` on a numeric or `ip` field, `IntPoint.newSetQuery`
/// and the like): the field, its dimensions and width, then the packed points.
pub const NODE_POINT_SET: u8 = 26;
const _: () = assert!(crate::queries::NODE_FIRST == NODE_POINT_SET + 1);
/// [`NODE_TERM_RANGE`]'s flags: which bounds follow, and which are inclusive.
pub const RANGE_HAS_LOWER: u8 = 1;
pub const RANGE_HAS_UPPER: u8 = 2;
pub const RANGE_INCLUDE_LOWER: u8 = 4;
pub const RANGE_INCLUDE_UPPER: u8 = 8;

/// Geometry tags inside the geo nodes ([`decode_geometry`]).
pub const GEOMETRY_POINT: u8 = 0;
pub const GEOMETRY_LINE: u8 = 1;
pub const GEOMETRY_POLYGON: u8 = 2;
pub const GEOMETRY_RECTANGLE: u8 = 3;
pub const GEOMETRY_CIRCLE: u8 = 4;

/// A request's query blob: the query, and the `min_score` in front of it
/// ([`QUERY_MIN_SCORE`]) when there is one -- rewritten, as
/// `IndexSearcher.search` rewrites before it builds a weight ([`rewritten`]);
/// and, before or after that `min_score`, the searcher's per-field
/// similarities ([`crate::similarity`]) when a field scores with another
/// than the default BM25. A request with them is always a boolean: the lone-term
/// search scores BM25.
pub fn decode_request(
    blob: &[u8],
) -> Result<(JvmQuery, Option<f32>, Option<PerFieldSimilarity>), Status> {
    use crate::similarity::QUERY_SIMILARITY;
    // The prefixes, in either order, each at most once.
    let (mut min, mut similarity) = (None, None);
    let mut c = Cursor { buf: blob, pos: 0 };
    loop {
        let at = c.pos;
        match blob.get(at) {
            Some(&QUERY_MIN_SCORE) if min.is_none() => {
                c.pos += 1;
                min = Some(f32::from_bits(u32::from_le_bytes(c.i32()?.to_le_bytes())));
            }
            Some(&QUERY_SIMILARITY) if similarity.is_none() => {
                c.pos += 1;
                similarity = Some(crate::similarity::decode(&mut c)?);
            }
            Some(&QUERY_MIN_SCORE | &QUERY_SIMILARITY) => {
                set_last_error("query blob: min_score or similarity table twice");
                return Err(Status::InvalidArgument);
            }
            _ => break,
        }
    }
    let rest = blob.get(c.pos..).ok_or(Status::InvalidArgument)?;
    let query = rewritten(decode_query(rest)?);
    Ok(match similarity {
        Some(s) => (JvmQuery::Boolean(boolean_of(&query)), min, Some(s)),
        None => (query, min, None),
    })
}

/// `IndexSearcher.rewrite`, before the free `search_*_multi_segment`
/// functions score a request: they take the query as written, and its shape
/// picks the scorer tree. An un-rewritten `#a ?(b c)` keeps its nested
/// disjunction as the lone scoring clause, which runs as a top-level WAND and
/// prunes to a different lower-bound total than Lucene's flattened
/// `#a ?b ?c` (177 vs 10,343 on `mixed_boolean_scoring_index`).
///
/// OpenSearch's `QueryPhase` hands the plugin a query that
/// `DefaultSearchContext.preProcess(true)` has already rewritten, and
/// `BooleanQuery.rewrite` is a fixpoint, so on that path this changes
/// nothing; it makes every other caller of the bridge score what Lucene
/// would score.
///
/// A query that rewrites to something other than a boolean (`?t` to `t`, a
/// pure negative one to `MatchNoDocsQuery`) becomes that boolean's one
/// `MUST` clause, as a non-boolean tree root always has: the boolean path
/// scores it exactly as the clause alone, and keeps what the term path
/// would not (a term's own `docFreq`, `NODE_TERM_STATS`).
pub fn rewritten(query: JvmQuery) -> JvmQuery {
    match query {
        JvmQuery::Term(t) => JvmQuery::Term(t),
        JvmQuery::Boolean(b) => match Clause::Boolean(Box::new(b)).rewrite() {
            Clause::Boolean(b) => JvmQuery::Boolean(*b),
            other => JvmQuery::Boolean(BooleanQuery {
                must: vec![other],
                ..Default::default()
            }),
        },
    }
}

/// `r`, a search's result over `sim`'s similarities, or the error a
/// scripted similarity's scorer kept (a `SimScorer` cannot fail): the
/// search's error, whatever else it returned.
pub fn similarity_checked<T>(
    sim: Option<&PerFieldSimilarity>,
    r: Result<T, Status>,
) -> Result<T, Status> {
    use lucene_search::similarities::Similarity;
    match sim.and_then(|s| s.take_failure()) {
        Some(e) => Err(crate::clauses::map_search_error(e)),
        None => r,
    }
}

/// One decoded query blob.
#[derive(Debug)]
pub enum JvmQuery {
    Term(TermQuery),
    Boolean(BooleanQuery),
}

/// A bounds-checked little-endian reader over a caller's blob. Every read
/// that would run past the end is [`Status::InvalidArgument`] -- the blob
/// is caller-supplied, so a short one is a bad argument, not a panic.
pub struct Cursor<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    /// The offset of the next read.
    pub fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Status> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let Some(end) = end else {
            set_last_error(format!(
                "query blob truncated: need {n} bytes at offset {}, have {}",
                self.pos,
                self.buf.len()
            ));
            return Err(Status::InvalidArgument);
        };
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// The bytes read since offset `start`.
    pub fn since(&self, start: usize) -> &'a [u8] {
        self.buf.get(start..self.pos).unwrap_or_default()
    }

    pub fn u8(&mut self) -> Result<u8, Status> {
        Ok(self.take(1)?[0])
    }

    pub fn i32(&mut self) -> Result<i32, Status> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn i64(&mut self) -> Result<i64, Status> {
        let b = self.take(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn f64(&mut self) -> Result<f64, Status> {
        Ok(f64::from_bits(self.i64()? as u64))
    }

    /// The bytes not read yet.
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub fn length(&mut self) -> Result<usize, Status> {
        let v = self.i32()?;
        usize::try_from(v).map_err(|_| {
            set_last_error(format!("query blob: negative length {v}"));
            Status::InvalidArgument
        })
    }

    pub fn bytes(&mut self) -> Result<&'a [u8], Status> {
        let n = self.length()?;
        self.take(n)
    }
}

/// Decodes a query blob. Little-endian throughout; every length is an `i32`
/// and must be non-negative.
///
/// | tag | layout after the tag byte |
/// |---|---|
/// | [`QUERY_TERM`] | `field_len`, field (UTF-8), `term_len`, term bytes |
/// | [`QUERY_BOOLEAN`] | `minimum_should_match`, `clause_count`, then per clause: `occur: u8`, `kind: u8`, `parent: i32`, `param: i32`, `field_len`, field, `term_len`, term |
///
/// A boolean clause's fields mean exactly what the parallel arrays of
/// [`read_boolean_clauses`] mean -- this decoder builds those arrays and hands
/// them over, so the nesting, depth and clause-count rules (and their error
/// messages) are that function's, not a second copy of them. A `BOOLEAN`
/// clause carries empty field and term bytes.
///
/// Trailing bytes are an error: a blob that decodes with bytes left over was
/// built by a writer that disagrees with this layout.
pub fn decode_query(blob: &[u8]) -> Result<JvmQuery, Status> {
    let mut c = Cursor { buf: blob, pos: 0 };
    let query = match c.u8()? {
        QUERY_TERM => {
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let term = c.bytes()?;
            JvmQuery::Term(TermQuery::new(field, term.to_vec()))
        }
        QUERY_BOOLEAN => {
            let msm = c.i32()?;
            let count = c.length()?;
            check_clause_count(count)?;
            let mut occurs = try_with_capacity::<u8>(count)?;
            let mut kinds = try_with_capacity::<u8>(count)?;
            let mut parents = try_with_capacity::<i32>(count)?;
            let mut params = try_with_capacity::<i32>(count)?;
            let mut spans = try_with_capacity::<(&[u8], &[u8])>(count)?;
            for _ in 0..count {
                occurs.push(c.u8()?);
                kinds.push(c.u8()?);
                parents.push(c.i32()?);
                params.push(c.i32()?);
                let field = c.bytes()?;
                let term = c.bytes()?;
                spans.push((field, term));
            }
            let term_at = |i: usize| {
                let (field, term) = spans[i];
                let field = std::str::from_utf8(field).map_err(|_| Status::InvalidUtf8)?;
                Ok((field, term))
            };
            let query = read_boolean_clauses(
                &occurs,
                &kinds,
                Some(&parents),
                Some(&params),
                &term_at,
                msm,
            )?;
            JvmQuery::Boolean(query)
        }
        QUERY_TREE => {
            let mut nodes = 0usize;
            match decode_node(&mut c, 0, &mut nodes)? {
                Clause::Boolean(b) => JvmQuery::Boolean(*b),
                other => JvmQuery::Boolean(BooleanQuery {
                    must: vec![other],
                    ..Default::default()
                }),
            }
        }
        other => {
            set_last_error(format!(
                "query blob: unknown query tag {other} (expected 0=TERM, 1=BOOLEAN, 2=TREE)"
            ));
            return Err(Status::InvalidArgument);
        }
    };
    if c.pos != blob.len() {
        set_last_error(format!(
            "query blob: {} trailing bytes after the query",
            blob.len() - c.pos
        ));
        return Err(Status::InvalidArgument);
    }
    Ok(query)
}

/// Whether `query` has a points clause anywhere, so the search opens the
/// segments' points (a cost a query without one should not pay).
pub fn query_uses_points(query: &JvmQuery) -> bool {
    match query {
        JvmQuery::Term(_) => false,
        JvmQuery::Boolean(b) => boolean_uses_points(b),
    }
}

/// [`query_uses_points`] of a decoded boolean.
pub fn boolean_uses_points(b: &BooleanQuery) -> bool {
    fn clause(c: &Clause) -> bool {
        match c {
            Clause::PointsRange(_) => true,
            Clause::Boolean(b) => boolean_uses_points(b),
            Clause::ConstantScore(c) => clause(&c.inner),
            Clause::Boost(b) => clause(&b.inner),
            Clause::DisjunctionMax(d) => d.disjuncts.iter().any(clause),
            // A points box or set reads the opened points (`exec::ranges`); a
            // geo node reads them through the segment's reader, unopened. A
            // join's or a function score's clauses may hold a range.
            Clause::Extended(e) => {
                matches!(
                    e.as_ref(),
                    lucene_search::extended_query::ExtendedQuery::PointRange(_)
                        | lucene_search::extended_query::ExtendedQuery::PointInSet(_)
                ) || e.children().into_iter().any(clause)
                    || crate::function_score::queries(e).into_iter().any(clause)
            }
            _ => false,
        }
    }
    b.must
        .iter()
        .chain(&b.filter)
        .chain(&b.should)
        .chain(&b.must_not)
        .any(clause)
}

/// One node of a [`QUERY_TREE`] blob, little-endian like the rest:
///
/// | kind | layout after the kind byte | Lucene query |
/// |---|---|---|
/// | `0` term | `field_len`, field (UTF-8), `term_len`, term | `TermQuery` |
/// | `1` boolean | `minimum_should_match: i32`, `count: i32`, then per clause `occur: u8` (0 `MUST`, 1 `FILTER`, 2 `SHOULD`, 3 `MUST_NOT`) and a node | `BooleanQuery` |
/// | `2` constant score | `score: f32` bits, node | `ConstantScoreQuery` (its score, normally 1) |
/// | `3` boost | `boost: f32` bits, node | `BoostQuery` |
/// | `4` dismax | `tie_breaker: f32` bits, `count: i32`, nodes | `DisjunctionMaxQuery` |
/// | `5` match all | nothing | `MatchAllDocsQuery` |
/// | `6` match none | nothing | `MatchNoDocsQuery` |
/// | `7` phrase | `field`, `slop: i32`, `count: i32`, then per term its position (`i32`) and bytes | `PhraseQuery` |
/// | `8` term set | `field`, `count: i32`, the terms | `TermInSetQuery` |
/// | `9` prefix | `field`, the prefix | `PrefixQuery` |
/// | `10` wildcard | `field`, the pattern (no `\` escapes) | `WildcardQuery` |
/// | `11` points range | `field`, `min: i64`, `max: i64` (sortable longs, inclusive) | a one-dimension 8-byte `PointRangeQuery` |
/// | `12` regexp | `field`, the pattern (UTF-8) | `RegexpQuery` with `RegExp.ALL` and no match flags |
/// | `13` exists | `field` | `FieldExistsQuery` |
/// | `14` term with its own statistics | `field`, `term`, `doc_freq: i64` (at least 1) | `TermQuery(term, TermStates)`: a `BlendedTermQuery`'s terms (fuzzy, `cross_fields`) |
/// | `15` points box | `field`, `num_dims: i32`, `lower`, `upper` (packed bytes, `num_dims` dimensions of equal width) | a `PointRangeQuery` of any width: `LatLonPoint.newBoxQuery`'s two dimensions |
/// | `16` geo distance | `field`, `lat: f64`, `lon: f64`, `radius_meters: f64` | `LatLonPointDistanceQuery` |
/// | `17` geo points | `field`, `relation: u8`, geometries ([`decode_geometries`]) | `LatLonPointQuery` (`newPolygonQuery`, `newGeometryQuery`) |
/// | `18` geo shapes | `field`, `relation: u8`, geometries | `LatLonShapeQuery` (`LatLonShape.newGeometryQuery`) |
/// | `19` geo shapes in a box | `field`, `relation: u8`, `min_lat`, `max_lat`, `min_lon`, `max_lon` (`f64`) | `LatLonShapeBoundingBoxQuery` (`LatLonShape.newBoxQuery`) |
/// | `20`-`24` | [`crate::nodes`] | block join, spans, intervals, combined fields, OpenSearch's function score |
///
/// A relation is `ShapeField.QueryRelation`'s ordinal: `0` `INTERSECTS`, `1`
/// `WITHIN`, `2` `DISJOINT`, `3` `CONTAINS`. The geo nodes (16-19) run as
/// [`DocumentClause`](lucene_search::extended_query::DocumentClause)s keyed
/// by their own bytes, so the query cache tells two of them apart exactly as
/// Java's `Query.equals` does (by every value each is built from). Their
/// geometries are validated as Java's constructors validate them; one Java
/// would refuse is an error here, not a query.
///
/// Depth is capped at `MAX_CLAUSE_DEPTH` and the whole tree at the clause
/// count limit, so the recursion is bounded by the blob, not trusted to it.
/// A score, boost or tie-breaker must be finite and non-negative (and a tie
/// breaker at most 1), as Lucene's constructors require.
pub fn decode_node(c: &mut Cursor<'_>, depth: usize, nodes: &mut usize) -> Result<Clause, Status> {
    use crate::clauses::MAX_CLAUSE_DEPTH;
    use lucene_search::query::{
        BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, FieldExistsQuery, MatchAllDocsQuery,
        MatchNoDocsQuery, PhraseQuery, PointsRangeQuery, PrefixQuery, RegexpQuery, TermInSetQuery,
        WildcardQuery,
    };
    if depth >= MAX_CLAUSE_DEPTH {
        set_last_error(format!(
            "query tree: nesting depth exceeds the maximum of {MAX_CLAUSE_DEPTH}"
        ));
        return Err(Status::InvalidArgument);
    }
    *nodes = nodes.saturating_add(1);
    check_clause_count(*nodes)?;
    let float = |c: &mut Cursor<'_>, what: &str, max: f32| -> Result<f32, Status> {
        let v = f32::from_bits(c.i32()? as u32);
        if !v.is_finite() || !(0.0..=max).contains(&v) {
            set_last_error(format!("query tree: {what} {v} is out of range"));
            return Err(Status::InvalidArgument);
        }
        Ok(v)
    };
    let start = c.pos;
    let kind = c.u8()?;
    Ok(match kind {
        NODE_TERM => {
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let term = c.bytes()?;
            Clause::Term(TermQuery::new(field, term.to_vec()))
        }
        NODE_BOOLEAN => {
            let msm = c.length()?;
            let count = c.length()?;
            check_clause_count(count)?;
            let mut b = BooleanQuery::new().with_minimum_should_match(msm);
            for _ in 0..count {
                let occur = c.u8()?;
                let child = decode_node(c, depth + 1, nodes)?;
                match occur {
                    0 => b.must.push(child),
                    1 => b.filter.push(child),
                    2 => b.should.push(child),
                    3 => b.must_not.push(child),
                    other => {
                        set_last_error(format!(
                            "query tree: unknown Occur tag {other} (expected 0..=3)"
                        ));
                        return Err(Status::InvalidArgument);
                    }
                }
            }
            Clause::Boolean(Box::new(b))
        }
        NODE_CONSTANT_SCORE => {
            let score = float(c, "constant score", f32::MAX)?;
            ConstantScoreQuery::new(decode_node(c, depth + 1, nodes)?, score).into()
        }
        NODE_BOOST => {
            let boost = float(c, "boost", f32::MAX)?;
            BoostQuery::new(decode_node(c, depth + 1, nodes)?, boost).into()
        }
        NODE_DISMAX => {
            let tie = float(c, "tie breaker", 1.0)?;
            let count = c.length()?;
            check_clause_count(count)?;
            let mut disjuncts = Vec::new();
            for _ in 0..count {
                disjuncts.push(decode_node(c, depth + 1, nodes)?);
            }
            DisjunctionMaxQuery::new(disjuncts, tie).into()
        }
        // The segment's own `maxDoc` is supplied per segment at search time
        // (`OpenSegment::max_doc`); the query carries none.
        NODE_MATCH_ALL => Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX)),
        NODE_MATCH_NONE => Clause::MatchNoDocs(MatchNoDocsQuery::new()),
        NODE_PHRASE => {
            // `field`, `slop`, then each term with its position. Only
            // consecutive positions from 0: the encoder falls a phrase with
            // gaps (a removed stopword) back to Lucene.
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let slop = c.length()?;
            let count = c.length()?;
            *nodes = nodes.saturating_add(count);
            check_clause_count(*nodes)?;
            if count == 0 {
                set_last_error("query tree: a phrase with no terms".to_string());
                return Err(Status::InvalidArgument);
            }
            // Grown as read, not sized from `count`: the blob bounds it.
            let mut terms = Vec::new();
            for i in 0..count {
                let position = c.i32()?;
                if usize::try_from(position).ok() != Some(i) {
                    set_last_error(format!(
                        "query tree: phrase term {i} is at position {position}, not {i}"
                    ));
                    return Err(Status::InvalidArgument);
                }
                terms.push(c.bytes()?.to_vec());
            }
            let slop = u32::try_from(slop).map_err(|_| Status::InvalidArgument)?;
            Clause::Phrase(PhraseQuery::new(field, terms).with_slop(slop))
        }
        NODE_TERM_SET => {
            // `TermInSetQuery`: `field`, then the terms, each counted against
            // the node limit as `IndexSearcher`'s clause visitor counts them.
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let count = c.length()?;
            *nodes = nodes.saturating_add(count);
            check_clause_count(*nodes)?;
            let mut terms = Vec::new();
            for _ in 0..count {
                terms.push(c.bytes()?.to_vec());
            }
            Clause::TermInSet(TermInSetQuery::new(field, terms))
        }
        NODE_PREFIX => {
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            Clause::Prefix(PrefixQuery::new(field, c.bytes()?.to_vec()))
        }
        NODE_WILDCARD => {
            // Lucene's syntax without its `\` escape, which the encoder never
            // sends (such a pattern falls back).
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            Clause::Wildcard(WildcardQuery::new(field, c.bytes()?.to_vec()))
        }
        NODE_REGEXP => {
            // `RegexpQuery(term)`'s grammar, `RegExp.ALL` and no match flags:
            // the encoder sends no other. Parsed here, so a pattern the native
            // grammar rejects fails the decode rather than the search.
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let pattern = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            lucene_codecs::regexp::RegexpPattern::new(pattern.as_bytes()).map_err(|e| {
                set_last_error(format!("query tree: regexp {pattern:?}: {e}"));
                Status::InvalidArgument
            })?;
            Clause::Regexp(RegexpQuery::new(field, pattern))
        }
        NODE_TERM_STATS => {
            // The term scores from the `docFreq` its `TermStates` carry
            // (blended across a fuzzy expansion or `cross_fields`' fields).
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let term = c.bytes()?.to_vec();
            let doc_freq = c.i64()?;
            if doc_freq < 1 {
                set_last_error(format!(
                    "query tree: a term's docFreq {doc_freq} is below 1"
                ));
                return Err(Status::InvalidArgument);
            }
            Clause::Term(TermQuery::new(field, term).with_doc_freq(doc_freq))
        }
        NODE_EXISTS => {
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            Clause::Exists(FieldExistsQuery::new(field))
        }
        NODE_TERM_RANGE => {
            // `field`, the flags byte, then the bounds it says are present (a
            // `null` bound in Java is an open end). `query_string`'s `[a TO
            // b]` and a `keyword` `range`, under `CONSTANT_SCORE_BLENDED_REWRITE`.
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let flags = c.u8()?;
            if flags
                & !(RANGE_HAS_LOWER | RANGE_HAS_UPPER | RANGE_INCLUDE_LOWER | RANGE_INCLUDE_UPPER)
                != 0
            {
                return Err(invalid(format!("query tree: term range flags {flags:#x}")));
            }
            let lower = if flags & RANGE_HAS_LOWER != 0 {
                Some(c.bytes()?.to_vec())
            } else {
                None
            };
            let upper = if flags & RANGE_HAS_UPPER != 0 {
                Some(c.bytes()?.to_vec())
            } else {
                None
            };
            let range = TermRangeQuery::new(
                field,
                lower,
                upper,
                flags & RANGE_INCLUDE_LOWER != 0,
                flags & RANGE_INCLUDE_UPPER != 0,
            );
            MultiTermQuery::new(
                MultiTermSource::TermRange(range),
                RewriteMethod::ConstantScoreBlended,
            )
            .into()
        }
        NODE_POINT_RANGE => {
            // A one-dimensional 8-byte `PointRangeQuery` (`long`, `date`,
            // `double`): its inclusive bounds as the sortable longs
            // `NumericUtils.sortableBytesToLong` reads from the packed bytes.
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let (min, max) = (c.i64()?, c.i64()?);
            Clause::PointsRange(PointsRangeQuery::new(field, min, max))
        }
        NODE_POINTS_BOX => {
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let num_dims = c.length()?;
            if !(1..=MAX_POINT_DIMS).contains(&num_dims) {
                set_last_error(format!(
                    "query tree: a points box of {num_dims} dimensions (want 1..={MAX_POINT_DIMS})"
                ));
                return Err(Status::InvalidArgument);
            }
            let lower = c.bytes()?.to_vec();
            let upper = c.bytes()?.to_vec();
            lucene_search::extended_query::PointRangeQuery::new(field, num_dims, lower, upper)
                .map_err(|e| invalid(format!("query tree: {e}")))?
                .into()
        }
        NODE_POINT_SET => {
            // One node, as `PointInSetQuery.visit` is one leaf to Lucene's
            // clause visitor; the points are checked and sorted by the
            // constructor.
            let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
            let num_dims = c.length()?;
            let bytes_per_dim = c.length()?;
            let count = c.length()?;
            let mut points = Vec::new();
            for _ in 0..count {
                points.push(c.bytes()?.to_vec());
            }
            lucene_search::extended_query::PointInSetQuery::new(
                field,
                num_dims,
                bytes_per_dim,
                points,
            )
            .map_err(|e| invalid(format!("query tree: {e}")))?
            .into()
        }
        NODE_GEO_DISTANCE | NODE_GEO_POINT | NODE_GEO_SHAPE | NODE_GEO_SHAPE_BOX => {
            return decode_geo(c, kind, start);
        }
        crate::nodes::NODE_TO_PARENT..=crate::nodes::NODE_LAST => {
            return crate::nodes::decode(c, kind, start, depth, nodes);
        }
        crate::queries::NODE_FIRST..=crate::queries::NODE_LAST => {
            return crate::queries::decode(c, kind, start, depth, nodes);
        }
        other => {
            set_last_error(format!(
                "query tree: unknown node kind {other} (expected 0..={})",
                crate::queries::NODE_LAST
            ));
            return Err(Status::InvalidArgument);
        }
    })
}

/// Lucene's `PointValues.MAX_DIMENSIONS`.
pub const MAX_POINT_DIMS: usize = 16;

/// [`Status::InvalidArgument`] with `message` as the last error.
pub fn invalid(message: String) -> Status {
    set_last_error(message);
    Status::InvalidArgument
}

/// A geo node (kinds 16-19, the kind byte already read; `start` is its
/// offset), as a [`DocumentClause`](lucene_search::extended_query::DocumentClause)
/// keyed by the node's bytes.
fn decode_geo(c: &mut Cursor<'_>, kind: u8, start: usize) -> Result<Clause, Status> {
    use lucene_search::document::geo::{
        LatLonPointDistanceQuery, LatLonPointQuery, LatLonShapeBoundingBoxQuery, LatLonShapeQuery,
        QueryRelation,
    };
    use lucene_search::document::DocumentQuery;
    use lucene_util::geo::Rectangle;
    let field = std::str::from_utf8(c.bytes()?).map_err(|_| Status::InvalidUtf8)?;
    let built = |e: lucene_search::Error| invalid(format!("query tree: geo query: {e}"));
    let relation = |c: &mut Cursor<'_>| -> Result<QueryRelation, Status> {
        Ok(match c.u8()? {
            0 => QueryRelation::Intersects,
            1 => QueryRelation::Within,
            2 => QueryRelation::Disjoint,
            3 => QueryRelation::Contains,
            other => {
                return Err(invalid(format!(
                    "query tree: unknown shape relation {other} (expected 0..=3)"
                )))
            }
        })
    };
    let query: std::sync::Arc<dyn DocumentQuery> = match kind {
        NODE_GEO_DISTANCE => {
            let (lat, lon, radius) = (c.f64()?, c.f64()?, c.f64()?);
            std::sync::Arc::new(
                LatLonPointDistanceQuery::new(field, lat, lon, radius).map_err(built)?,
            )
        }
        NODE_GEO_POINT => {
            let rel = relation(c)?;
            let geometries = decode_geometries(c)?;
            std::sync::Arc::new(LatLonPointQuery::new(field, rel, &geometries).map_err(built)?)
        }
        NODE_GEO_SHAPE => {
            let rel = relation(c)?;
            let geometries = decode_geometries(c)?;
            std::sync::Arc::new(LatLonShapeQuery::new(field, rel, &geometries).map_err(built)?)
        }
        _ => {
            let rel = relation(c)?;
            let (min_lat, max_lat, min_lon, max_lon) = (c.f64()?, c.f64()?, c.f64()?, c.f64()?);
            let rect = Rectangle::new(min_lat, max_lat, min_lon, max_lon)
                .map_err(|e| invalid(format!("query tree: geo query: {e}")))?;
            std::sync::Arc::new(LatLonShapeBoundingBoxQuery::new(field, rel, rect).map_err(built)?)
        }
    };
    let key = hex(&c.buf[start..c.pos]);
    Ok(lucene_search::extended_query::DocumentClause::new(key, query).into())
}

/// `bytes` as lowercase hex: a geo node's query-cache key.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::new();
    // The bytes are the blob's own, already in memory.
    out.reserve(bytes.len().saturating_mul(2));
    for &b in bytes {
        out.push(char::from(DIGITS[usize::from(b >> 4)]));
        out.push(char::from(DIGITS[usize::from(b & 15)]));
    }
    out
}

/// A geo node's geometries: `count: i32` (at least 1), then each a tag and
/// its coordinates (`f64`s):
///
/// | tag | layout | Lucene |
/// |---|---|---|
/// | `0` point | `lat`, `lon` | `Point` |
/// | `1` line | `n: i32`, `n` lats, `n` lons | `Line` |
/// | `2` polygon | `n: i32`, `n` lats, `n` lons, `holes: i32`, each hole a polygon without holes | `Polygon` |
/// | `3` rectangle | `min_lat`, `max_lat`, `min_lon`, `max_lon` | `Rectangle` |
/// | `4` circle | `lat`, `lon`, `radius_meters` | `Circle` |
///
/// Each is built through its constructor, which checks what Java's does.
/// Counts are bounded by the blob (every vertex is 16 bytes of it).
fn decode_geometries(c: &mut Cursor<'_>) -> Result<Vec<lucene_util::geo::LatLonGeometry>, Status> {
    let count = c.length()?;
    if count == 0 {
        return Err(invalid("query tree: a geo query with no geometries".into()));
    }
    let mut out = Vec::new();
    for _ in 0..count {
        out.push(decode_geometry(c)?);
    }
    Ok(out)
}

/// One geometry of [`decode_geometries`].
fn decode_geometry(c: &mut Cursor<'_>) -> Result<lucene_util::geo::LatLonGeometry, Status> {
    use lucene_util::geo::{Circle, LatLonGeometry, Line, Point, Rectangle};
    let geo = |e: lucene_util::geo::GeoError| invalid(format!("query tree: geometry: {e}"));
    Ok(match c.u8()? {
        GEOMETRY_POINT => {
            let (lat, lon) = (c.f64()?, c.f64()?);
            LatLonGeometry::Point(Point::new(lat, lon).map_err(geo)?)
        }
        GEOMETRY_LINE => {
            let (lats, lons) = decode_vertices(c)?;
            LatLonGeometry::Line(Line::new(&lats, &lons).map_err(geo)?)
        }
        GEOMETRY_POLYGON => LatLonGeometry::Polygon(decode_polygon(c, true)?),
        GEOMETRY_RECTANGLE => {
            let (a, b, d, e) = (c.f64()?, c.f64()?, c.f64()?, c.f64()?);
            LatLonGeometry::Rectangle(Rectangle::new(a, b, d, e).map_err(geo)?)
        }
        GEOMETRY_CIRCLE => {
            let (lat, lon, radius) = (c.f64()?, c.f64()?, c.f64()?);
            LatLonGeometry::Circle(Circle::new(lat, lon, radius).map_err(geo)?)
        }
        other => {
            return Err(invalid(format!(
                "query tree: unknown geometry tag {other} (expected 0..=4)"
            )))
        }
    })
}

/// A polygon of [`decode_geometry`]; a hole (`holes == false`) may have
/// none of its own, as Java's `Polygon` requires.
fn decode_polygon(c: &mut Cursor<'_>, holes: bool) -> Result<lucene_util::geo::Polygon, Status> {
    let (lats, lons) = decode_vertices(c)?;
    let n = c.length()?;
    if n > 0 && !holes {
        return Err(invalid("query tree: a polygon hole with holes".into()));
    }
    let mut inner = Vec::new();
    for _ in 0..n {
        inner.push(decode_polygon(c, false)?);
    }
    lucene_util::geo::Polygon::new(&lats, &lons, inner)
        .map_err(|e| invalid(format!("query tree: geometry: {e}")))
}

/// `n`, then `n` latitudes and `n` longitudes; `n` is checked against the
/// bytes left before anything is allocated.
fn decode_vertices(c: &mut Cursor<'_>) -> Result<(Vec<f64>, Vec<f64>), Status> {
    let n = c.length()?;
    let need = n.checked_mul(16).ok_or(Status::InvalidArgument)?;
    if need > c.buf.len().saturating_sub(c.pos) {
        return Err(invalid(format!(
            "query tree: {n} vertices, but only {} bytes are left",
            c.buf.len().saturating_sub(c.pos)
        )));
    }
    let mut lats = try_with_capacity::<f64>(n)?;
    for _ in 0..n {
        lats.push(c.f64()?);
    }
    let mut lons = try_with_capacity::<f64>(n)?;
    for _ in 0..n {
        lons.push(c.f64()?);
    }
    Ok((lats, lons))
}

/// A decoded query as the boolean the search functions take (a lone term as
/// its one `MUST` clause).
pub fn boolean_of(query: &JvmQuery) -> BooleanQuery {
    match query {
        JvmQuery::Term(t) => BooleanQuery {
            must: vec![Clause::Term(t.clone())],
            ..Default::default()
        },
        JvmQuery::Boolean(b) => b.clone(),
    }
}

/// A query-blob writer for tests that drive the entry points.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::QUERY_TREE;

    /// A little-endian blob, as the Java encoders write one.
    #[derive(Default, Clone)]
    pub struct Blob(pub Vec<u8>);

    impl Blob {
        pub fn tree() -> Self {
            Blob(vec![QUERY_TREE])
        }
        pub fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }
        pub fn i32(mut self, v: i32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        pub fn f32(self, v: f32) -> Self {
            self.i32(v.to_bits() as i32)
        }
        pub fn f64(mut self, v: f64) -> Self {
            self.0.extend_from_slice(&v.to_bits().to_le_bytes());
            self
        }
        pub fn bytes(self, b: &[u8]) -> Self {
            let mut s = self.i32(b.len() as i32);
            s.0.extend_from_slice(b);
            s
        }
        pub fn str(self, s: &str) -> Self {
            self.bytes(s.as_bytes())
        }
        pub fn raw(mut self, b: &Blob) -> Self {
            self.0.extend_from_slice(&b.0);
            self
        }
        /// A term node.
        pub fn term(self, field: &str, t: &str) -> Self {
            self.u8(0).str(field).str(t)
        }
        /// An exists node.
        pub fn exists(self, field: &str) -> Self {
            self.u8(13).str(field)
        }
    }
}
