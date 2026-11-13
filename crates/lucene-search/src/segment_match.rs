//! Matching a query segment by segment for the code that reads a segment's
//! columns over the matches: a segment's plain (unscored) context, its live
//! matches (or "every live document", which a column is then streamed for),
//! the scored matches `min_score` keeps, and how a column is read over them.

use std::collections::HashMap;

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::exec::{self, Mode};
use crate::field_norms::FieldNorms;
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause};
use crate::Result;

pub use crate::exec::LeafContext;

/// OpenSearch's `min_score` in front of the aggregations: its
/// `MinimumScoreCollector` wraps them too, so a document counts only when it
/// scores at least `min` -- the query scored in `COMPLETE` mode with the
/// segments' norms (per segment, parallel to the segments) and the reader's
/// statistics.
pub struct MinScore<'a, 'n> {
    pub min: f32,
    pub norms: &'a [Option<&'a HashMap<String, FieldNorms<'n>>>],
}

/// A segment's context for matching without scores (no norms, no
/// statistics).
pub fn plain_context<'s>(seg: &'s OpenSegment<'_>) -> exec::LeafContext<'s> {
    exec::LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: seg.live_docs,
        points: seg.points,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: seg.cache,
        reader: seg.reader,
        similarity: seg.similarity,
    }
}

/// A segment's live matches of `clause`: `None` when it has none (no
/// scorer), `Some(None)` for every live document (a match-all, which the
/// caller reads straight down its columns), else the documents, collected
/// into `buf` once.
pub fn segment_matches<'b>(
    ctx: &exec::LeafContext<'_>,
    query: &BooleanQuery,
    clause: &Clause,
    live: Option<&FixedBitSet>,
    buf: &'b mut Vec<i32>,
) -> Result<Option<Option<&'b [i32]>>> {
    if matches_everything(clause) {
        return Ok(Some(None));
    }
    // The bulk scorer, as `IndexSearcher.search` collects an aggregation:
    // a disjunction a window at a time, not a heap step per document.
    struct Docs<'v>(&'v mut Vec<i32>);
    impl crate::collector::ScoringCollector for Docs<'_> {
        fn collect(&mut self, doc_id: i32, _score: f32) {
            self.0.push(doc_id);
        }
        fn score_mode(&self) -> crate::collector::ScoreMode {
            crate::collector::ScoreMode::CompleteNoScores
        }
    }
    let Some(mut bulk) = exec::bulk_boolean(ctx, query, 1.0, Mode::NoScores)? else {
        return Ok(None);
    };
    buf.clear();
    exec::score_segment(&mut bulk, Mode::NoScores, live, &mut Docs(buf), ctx.max_doc)?;
    // The readers downstream walk forward and count each document once; a
    // bulk scorer hands documents out strictly ascending, and this keeps that
    // a checked fact rather than an assumption.
    if !buf.windows(2).all(|w| w[0] < w[1]) {
        buf.sort_unstable();
        buf.dedup();
    }
    Ok(Some(Some(&buf[..])))
}

/// [`segment_matches`] read as [`column_read`] reads them, in one pass over
/// the matches: the bulk scorer marks each match in the segment's bit set
/// as it collects it (no list of the matches is built and walked again),
/// and the matches are listed from the bits only when they are sparse
/// enough to seek. The same documents, so the same reads, as
/// `column_read(segment_matches(..))`; also the live match count. `None`
/// when the query matches nothing in the segment.
#[allow(clippy::too_many_arguments)]
pub fn segment_match_read<'b>(
    ctx: &exec::LeafContext<'_>,
    query: &BooleanQuery,
    clause: &Clause,
    live: Option<&'b FixedBitSet>,
    max_doc: i32,
    words: &'b mut Vec<u64>,
    docs: &'b mut Vec<i32>,
) -> Result<Option<(ColumnRead<'b>, u64)>> {
    let max = usize::try_from(max_doc).unwrap_or(0);
    if matches_everything(clause) {
        let count = live.map_or(max, FixedBitSet::cardinality);
        return Ok(Some((
            ColumnRead::Stream(Accept::Live(live)),
            u64::try_from(count).unwrap_or(u64::MAX),
        )));
    }
    let Some(mut bulk) = exec::bulk_boolean(ctx, query, 1.0, Mode::NoScores)? else {
        return Ok(None);
    };
    words.clear();
    words.resize(max.div_ceil(64), 0);
    if !copy_constant_bits(&bulk, live, words) {
        exec::score_segment(
            &mut bulk,
            Mode::NoScores,
            live,
            &mut Marks(words),
            ctx.max_doc,
        )?;
    }
    let count: u64 = words.iter().map(|w| u64::from(w.count_ones())).sum();
    // As `column_read`: seek when the matches are at most a quarter of the
    // segment.
    if usize::try_from(count)
        .unwrap_or(usize::MAX)
        .saturating_mul(4)
        <= max
    {
        docs.clear();
        for (w, &word) in words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                // ARITH: `w < max_doc / 64 + 1` and the bit `< 64`: a
                // document below `max_doc`, an `i32`.
                #[allow(clippy::arithmetic_side_effects)]
                docs.push((w * 64) as i32 + bits.trailing_zeros() as i32);
                bits &= bits - 1;
            }
        }
        return Ok(Some((ColumnRead::Seek(docs), count)));
    }
    Ok(Some((ColumnRead::Stream(Accept::Marked(words)), count)))
}

/// A bulk scorer's matches marked in a segment's bit set.
struct Marks<'w>(&'w mut [u64]);
impl crate::collector::ScoringCollector for Marks<'_> {
    #[inline]
    fn collect(&mut self, doc_id: i32, _score: f32) {
        let d = doc_id as u32 as usize;
        if let Some(w) = self.0.get_mut(d >> 6) {
            *w |= 1 << (d & 63);
        }
    }
    fn score_mode(&self) -> crate::collector::ScoreMode {
        crate::collector::ScoreMode::CompleteNoScores
    }
    /// No threshold: a batch is marked whole.
    /// The batch's bits gathered in a register while its documents stay
    /// in one word (ascending, they mostly do), each word stored once:
    /// a store per document would wait on the previous one's.
    fn collect_many(&mut self, docs: &[i32], _scores: &[f32], doc_base: i32) -> usize {
        let mut at = usize::MAX;
        let mut bits = 0u64;
        for &d in docs {
            let d = d.saturating_add(doc_base) as u32 as usize;
            if d >> 6 != at {
                if let Some(w) = self.0.get_mut(at) {
                    *w |= bits;
                }
                at = d >> 6;
                bits = 0;
            }
            bits |= 1 << (d & 63);
        }
        if let Some(w) = self.0.get_mut(at) {
            *w |= bits;
        }
        docs.len()
    }
}

/// [`segment_matches`] as the segment's bit set (`words`, one bit per
/// document, live ones only): `None` when the query matches nothing in the
/// segment, `Some(false)` when it matches every live document (`words`
/// untouched), `Some(true)` when `words` holds the matches. The same
/// documents as [`segment_matches`], in 1 bit a document instead of 32 a
/// match.
pub fn segment_match_bits(
    ctx: &exec::LeafContext<'_>,
    query: &BooleanQuery,
    clause: &Clause,
    live: Option<&FixedBitSet>,
    max_doc: i32,
    words: &mut Vec<u64>,
) -> Result<Option<bool>> {
    if matches_everything(clause) {
        return Ok(Some(false));
    }
    let Some(mut bulk) = exec::bulk_boolean(ctx, query, 1.0, Mode::NoScores)? else {
        return Ok(None);
    };
    let max = usize::try_from(max_doc).unwrap_or(0);
    words.clear();
    words.resize(max.div_ceil(64), 0);
    if !copy_constant_bits(&bulk, live, words) {
        exec::score_segment(
            &mut bulk,
            Mode::NoScores,
            live,
            &mut Marks(words),
            ctx.max_doc,
        )?;
    }
    Ok(Some(true))
}

/// The marks of a bulk scorer over a constant-scored bit set (a points range
/// or a cached clause: what `DefaultBulkScorer` hands over as a
/// `DocIdStream`) taken a word at a time -- the set's words from the
/// scorer's document on, and-ed with the live ones -- instead of one
/// collected document at a time. The same documents [`exec::score_segment`]
/// would collect. `false` (nothing written) for any other bulk scorer.
fn copy_constant_bits(
    bulk: &exec::Bulk<'_>,
    live: Option<&FixedBitSet>,
    words: &mut [u64],
) -> bool {
    let exec::Bulk::Scorer(scorer, _) = bulk else {
        return false;
    };
    if scorer.two_phase() {
        return false;
    }
    let Some((set, _)) = scorer.constant_bits() else {
        return false;
    };
    let exec::cache::CachedSet::Bits { bits, .. } = &*set else {
        return false;
    };
    // The scorer's document (unpositioned: -1) is its first match; nothing
    // before it is collected.
    let from = usize::try_from(scorer.doc_id()).unwrap_or(0);
    let live_words = live.map(FixedBitSet::words);
    for (i, (w, &b)) in words.iter_mut().zip(bits.words()).enumerate() {
        // A word past the live set's is no live document's.
        let mut b = b & live_words.map_or(u64::MAX, |lw| lw.get(i).copied().unwrap_or(0));
        // ARITH: `i < words.len()`, a word index of the segment.
        #[allow(clippy::arithmetic_side_effects)]
        if i << 6 < from {
            b &= if (i + 1) << 6 <= from {
                0
            } else {
                u64::MAX << (from & 63)
            };
        }
        *w = b;
    }
    true
}

/// [`segment_matches`] behind `min_score`: the matches scoring at least
/// `min`, the query scored in `COMPLETE` mode (never a match-all read straight
/// down the columns).
pub fn segment_matches_scoring<'b>(
    ctx: &exec::LeafContext<'_>,
    query: &BooleanQuery,
    live: Option<&FixedBitSet>,
    min: f32,
    buf: &'b mut Vec<i32>,
) -> Result<Option<Option<&'b [i32]>>> {
    struct Docs<'v>(&'v mut Vec<i32>);
    impl crate::collector::ScoringCollector for Docs<'_> {
        fn collect(&mut self, doc_id: i32, _score: f32) {
            self.0.push(doc_id);
        }
    }
    let Some(mut bulk) = exec::bulk_boolean(ctx, query, 1.0, Mode::Complete)? else {
        return Ok(None);
    };
    buf.clear();
    let mut docs = Docs(buf);
    let mut passing = crate::collector::MinScoreCollector::new(&mut docs, min);
    exec::score_segment(&mut bulk, Mode::Complete, live, &mut passing, ctx.max_doc)?;
    if !buf.windows(2).all(|w| w[0] < w[1]) {
        buf.sort_unstable();
        buf.dedup();
    }
    Ok(Some(Some(&buf[..])))
}

/// [`segment_matches_scoring`] keeping each match's score too, a range of
/// documents at a time (`BulkScorer.score(collector, acceptDocs, min, max)`
/// over consecutive windows): what a pass reading scores holds is one
/// range's matches, not the segment's.
pub struct ScoredRanges<'a> {
    bulk: Option<exec::Bulk<'a>>,
    live: Option<&'a FixedBitSet>,
    min: f32,
    /// The first document the next range scores.
    next: i32,
}

impl<'a> ScoredRanges<'a> {
    /// `query` over the segment of `ctx` from document `from` on, scored in
    /// `COMPLETE` mode behind `min_score` `min`.
    ///
    /// # Errors
    /// What building the scorer reports.
    pub fn open(
        ctx: &exec::LeafContext<'a>,
        query: &BooleanQuery,
        live: Option<&'a FixedBitSet>,
        min: f32,
        from: i32,
    ) -> Result<Self> {
        Ok(ScoredRanges {
            bulk: exec::bulk_boolean(ctx, query, 1.0, Mode::Complete)?,
            live,
            min,
            next: from,
        })
    }

    /// The matches of the next range of `width` documents below `end`, by
    /// document, in `docs` and their scores in `scores`; `false` once
    /// nothing is left to score below `end`. A range may hold no match.
    ///
    /// # Errors
    /// What scoring reports.
    pub fn next(
        &mut self,
        width: i32,
        end: i32,
        docs: &mut Vec<i32>,
        scores: &mut Vec<f32>,
    ) -> Result<bool> {
        struct Hits<'v>(&'v mut Vec<i32>, &'v mut Vec<f32>);
        impl crate::collector::ScoringCollector for Hits<'_> {
            fn collect(&mut self, doc_id: i32, score: f32) {
                self.0.push(doc_id);
                self.1.push(score);
            }
        }
        docs.clear();
        scores.clear();
        let Some(bulk) = self.bulk.as_mut() else {
            return Ok(false);
        };
        if self.next >= end {
            return Ok(false);
        }
        let max = self.next.saturating_add(width).min(end);
        let mut sink = Hits(docs, scores);
        let mut passing = crate::collector::MinScoreCollector::new(&mut sink, self.min);
        self.next = bulk.score(Mode::Complete, self.live, &mut passing, self.next, max)?;
        if !docs.windows(2).all(|w| w[0] < w[1]) {
            let mut hits: Vec<(i32, f32)> =
                docs.iter().copied().zip(scores.iter().copied()).collect();
            hits.sort_by_key(|h| h.0);
            hits.dedup_by_key(|h| h.0);
            docs.clear();
            scores.clear();
            docs.extend(hits.iter().map(|h| h.0));
            scores.extend(hits.iter().map(|h| h.1));
        }
        Ok(true)
    }
}

/// How a segment's matches meet a column.
pub enum ColumnRead<'b> {
    /// Stream the column; a document counts when `accept` says so: every
    /// live one (a match-all), or one marked in a bit set of the matches
    /// (when they are dense enough that one pass down the column beats a
    /// seek per match).
    Stream(Accept<'b>),
    /// Seek each match's values.
    Seek(&'b [i32]),
}

/// Which documents a streamed column read keeps.
pub enum Accept<'b> {
    Live(Option<&'b FixedBitSet>),
    Marked(&'b [u64]),
}

impl Accept<'_> {
    /// Word `i` of the accepted documents' bits: documents `64 i..64 i + 64`.
    #[inline]
    pub fn word(&self, i: usize) -> u64 {
        match self {
            Accept::Live(None) => u64::MAX,
            Accept::Live(Some(l)) => l.words().get(i).copied().unwrap_or(0),
            Accept::Marked(words) => words.get(i).copied().unwrap_or(0),
        }
    }

    #[inline]
    pub fn test(&self, doc: i32) -> bool {
        match self {
            Accept::Live(live) => live.is_none_or(|l| l.get_doc(doc)),
            Accept::Marked(words) => {
                let d = doc as u32 as usize;
                words.get(d >> 6).is_some_and(|w| w >> (d & 63) & 1 == 1)
            }
        }
    }
}

/// Streams a column when the matches cover more than a quarter of the
/// segment -- one decode per stored value and a bit test per document is
/// then cheaper than a seek per match -- else seeks. `words` holds the bit
/// set when one is built.
pub fn column_read<'b>(
    docs: Option<&'b [i32]>,
    live: Option<&'b FixedBitSet>,
    max_doc: i32,
    words: &'b mut Vec<u64>,
) -> ColumnRead<'b> {
    let Some(docs) = docs else {
        return ColumnRead::Stream(Accept::Live(live));
    };
    let max = usize::try_from(max_doc).unwrap_or(0);
    if docs.len().saturating_mul(4) <= max {
        return ColumnRead::Seek(docs);
    }
    words.clear();
    words.resize(max.div_ceil(64), 0);
    for &d in docs {
        let d = d as u32 as usize;
        if let Some(w) = words.get_mut(d >> 6) {
            *w |= 1 << (d & 63);
        }
    }
    ColumnRead::Stream(Accept::Marked(words))
}

/// Whether `clause` matches every document, whatever wraps the match-all
/// (a constant score, a boost, a boolean whose required clauses all match
/// everything): its matches are then every live document, in order, and the
/// columns can be read straight down.
fn matches_everything(clause: &Clause) -> bool {
    match clause {
        Clause::MatchAllDocs(_) => true,
        Clause::ConstantScore(c) => matches_everything(&c.inner),
        Clause::Boost(b) => matches_everything(&b.inner),
        Clause::Boolean(b) if b.must_not.is_empty() => {
            let required: Vec<&Clause> = b.must.iter().chain(&b.filter).collect();
            if required.is_empty() {
                b.minimum_should_match <= 1 && b.should.iter().any(matches_everything)
            } else {
                b.minimum_should_match == 0 && required.into_iter().all(matches_everything)
            }
        }
        _ => false,
    }
}

/// `BooleanQuery.rewrite`: a boolean of one clause is that clause.
pub fn lone_clause(query: &BooleanQuery) -> Clause {
    let clauses = query.must.len() + query.filter.len() + query.should.len() + query.must_not.len();
    match (&query.must[..], &query.filter[..], &query.should[..]) {
        ([only], _, _) if clauses == 1 && query.minimum_should_match == 0 => only.clone(),
        (_, [only], _) if clauses == 1 && query.minimum_should_match == 0 => only.clone(),
        (_, _, [only]) if clauses == 1 && query.minimum_should_match <= 1 => only.clone(),
        _ => Clause::Boolean(Box::new(query.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The one-pass match read marks what collecting the matches marks.**
    /// A dense range over the terms fixture (deletions in three of its four
    /// segments) is a constant-scored bit set, copied a word at a time;
    /// with the scorer moved past its first documents only the rest are
    /// copied; a term goes through the bulk scorer's batches.
    #[test]
    fn one_pass_match_reads_mark_what_collecting_marks() {
        let dir = lucene_store::FsDirectory::open(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/terms_aggs_index"
        )));
        let reader = crate::directory_reader::DirectoryReader::open(&dir).unwrap();
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let filter = |c: Clause| BooleanQuery {
            filter: vec![c],
            ..Default::default()
        };
        let queries = [
            filter(Clause::PointsRange(crate::query::PointsRangeQuery::new(
                "r",
                0,
                i64::MAX,
            ))),
            filter(Clause::Term(crate::query::TermQuery::new(
                "body",
                b"w0".to_vec(),
            ))),
        ];
        let mut copied = 0;
        for seg in &segments {
            let ctx = exec::LeafContext {
                fields: seg.fields,
                doc_in: seg.doc_in,
                pos_in: seg.pos_in,
                pay_in: seg.pay_in,
                live_docs: seg.live_docs,
                points: seg.points,
                norms: None,
                global: None,
                max_doc: seg.max_doc,
                cache: seg.cache,
                reader: seg.reader,
                similarity: seg.similarity,
            };
            let max_doc = seg.reader.unwrap().max_doc;
            for q in &queries {
                let clause = lone_clause(q);
                let (mut words, mut docs, mut buf, mut want_words) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let want = segment_matches(&ctx, q, &clause, seg.live_docs, &mut buf).unwrap();
                let got = segment_match_read(
                    &ctx,
                    q,
                    &clause,
                    seg.live_docs,
                    max_doc,
                    &mut words,
                    &mut docs,
                )
                .unwrap();
                let (Some(want), Some((got, count))) = (want, got) else {
                    panic!("{q:?} matches in every segment");
                };
                let want_docs: Vec<i32> = want.unwrap().to_vec();
                assert_eq!(count, want_docs.len() as u64);
                match (
                    got,
                    column_read(Some(&want_docs), seg.live_docs, max_doc, &mut want_words),
                ) {
                    (ColumnRead::Seek(g), ColumnRead::Seek(w)) => assert_eq!(g, w),
                    (
                        ColumnRead::Stream(Accept::Marked(g)),
                        ColumnRead::Stream(Accept::Marked(w)),
                    ) => {
                        assert_eq!(g, w)
                    }
                    _ => panic!("{q:?}: different reads"),
                }
                // The bit set copied with its scorer moved on: only the
                // documents from there are marked.
                let mut bulk = exec::bulk_boolean(&ctx, q, 1.0, Mode::NoScores)
                    .unwrap()
                    .unwrap();
                let exec::Bulk::Scorer(scorer, _) = &mut bulk else {
                    continue;
                };
                let target = scorer.advance(70).unwrap();
                let mut marks = vec![0u64; usize::try_from(max_doc).unwrap().div_ceil(64)];
                if !copy_constant_bits(&bulk, seg.live_docs, &mut marks) {
                    continue;
                }
                copied += 1;
                for &d in &want_docs {
                    let set = marks[d as usize >> 6] >> (d & 63) & 1 == 1;
                    assert_eq!(set, d >= target, "doc {d}, scorer on {target}");
                }
                let marked: u32 = marks.iter().map(|w| w.count_ones()).sum();
                assert_eq!(
                    marked as usize,
                    want_docs.iter().filter(|&&d| d >= target).count()
                );
            }
        }
        assert!(copied > 0, "no segment's range was a bit set");
        // Any other bulk scorer is not copied.
        let ctx_seg = &segments[0];
        let ctx = exec::LeafContext {
            fields: ctx_seg.fields,
            doc_in: ctx_seg.doc_in,
            pos_in: ctx_seg.pos_in,
            pay_in: ctx_seg.pay_in,
            live_docs: ctx_seg.live_docs,
            points: ctx_seg.points,
            norms: None,
            global: None,
            max_doc: ctx_seg.max_doc,
            cache: ctx_seg.cache,
            reader: ctx_seg.reader,
            similarity: ctx_seg.similarity,
        };
        let bulk = exec::bulk_boolean(&ctx, &queries[1], 1.0, Mode::NoScores)
            .unwrap()
            .unwrap();
        assert!(!copy_constant_bits(&bulk, None, &mut [0u64; 4]));
    }

    #[test]
    fn a_match_all_is_seen_through_its_wrappers() {
        let all = || Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(0));
        let term = || Clause::Term(crate::TermQuery::new("f", b"x".to_vec()));
        assert!(matches_everything(&all()));
        assert!(matches_everything(&Clause::ConstantScore(Box::new(
            crate::query::ConstantScoreQuery::new(all(), 1.0)
        ))));
        assert!(matches_everything(&Clause::Boost(Box::new(
            crate::query::BoostQuery::new(all(), 2.0)
        ))));
        let mut b = BooleanQuery::new();
        b.filter.push(all());
        b.must.push(all());
        assert!(matches_everything(&Clause::Boolean(Box::new(b.clone()))));
        b.must_not.push(term());
        assert!(!matches_everything(&Clause::Boolean(Box::new(b))));
        let mut b = BooleanQuery::new();
        b.should.push(term());
        b.should.push(all());
        assert!(matches_everything(&Clause::Boolean(Box::new(b.clone()))));
        b.minimum_should_match = 2;
        assert!(!matches_everything(&Clause::Boolean(Box::new(b))));
        let mut b = BooleanQuery::new();
        b.must.push(all());
        b.filter.push(term());
        assert!(!matches_everything(&Clause::Boolean(Box::new(b))));
        assert!(!matches_everything(&term()));
    }
}
