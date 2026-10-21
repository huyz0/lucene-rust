//! `OffsetsEnum.OfMatchesIteratorWithSubs`: the offsets of a field's
//! `MatchesIterator`, descending into the sub-matches of a match that has
//! them (a phrase or an interval reports its terms, not itself).

use std::sync::{Arc, Mutex, PoisonError};

use crate::matches::BoxMatchesIterator;
use crate::query::Clause;
use crate::query_visitor::{self, QueryVisitor, Term};

use super::super::Result;
use super::passage::OffsetsEnum;

/// `queryToTerm`: the query's terms joined by spaces, else its `toString()`.
pub(super) fn query_to_term(query: &Clause) -> Vec<u8> {
    struct Terms(Vec<u8>);
    impl QueryVisitor for Terms {
        fn consume_terms(&mut self, _query: &Clause, terms: &[Term]) {
            for t in terms {
                if !self.0.is_empty() {
                    self.0.push(b' ');
                }
                self.0.extend_from_slice(&t.bytes);
            }
        }
    }
    let mut v = Terms(Vec::new());
    query_visitor::visit(query, &mut v);
    if v.0.is_empty() {
        crate::explain::describe_clause(query).into_bytes()
    } else {
        v.0
    }
}

enum Slot {
    /// `OfMatchesIterator`.
    Iterator(BoxMatchesIterator),
    /// `CachedOE`.
    Cached,
}

struct Pending {
    slot: Slot,
    term: Vec<u8>,
    start: i32,
    end: i32,
}

/// How many queries' [`query_to_term`] a [`QueryTerms`] remembers.
const REMEMBERED_TERMS: usize = 64;

/// [`query_to_term`] of the queries met so far, shared by the enums of a
/// strategy: matches report the same few queries over and over, and naming
/// a multi-term one visits it -- compiling its automaton -- only to find it
/// has no single term, so its `toString()` stands in.
#[derive(Default)]
pub(super) struct QueryTerms(Mutex<Vec<(Clause, Vec<u8>)>>);

impl QueryTerms {
    /// [`query_to_term`], remembered.
    fn term_of(&self, query: &Clause) -> Vec<u8> {
        if let Clause::Term(t) = query {
            // the one term (an empty one is the query's `toString()`)
            if !t.term.is_empty() {
                return t.term.clone();
            }
        }
        // a poisoned lock holds terms that are still right
        let mut known = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, term)) = known.iter().find(|(q, _)| q == query) {
            return term.clone();
        }
        let term = query_to_term(query);
        if known.len() < REMEMBERED_TERMS {
            known.push((query.clone(), term.clone()));
        }
        term
    }
}

/// `OffsetsEnum.OfMatchesIteratorWithSubs`.
pub struct OfMatchesWithSubs {
    /// `pendingQueue`, ascending by (start, end, term).
    pending: Vec<Pending>,
    terms: Arc<QueryTerms>,
}

impl OfMatchesWithSubs {
    /// `new OfMatchesIteratorWithSubs(matchesIterator)`.
    pub fn new(iterator: BoxMatchesIterator) -> Self {
        Self::with_terms(iterator, Arc::default())
    }

    /// [`Self::new`], naming the queries' terms through `terms`.
    pub(super) fn with_terms(iterator: BoxMatchesIterator, terms: Arc<QueryTerms>) -> Self {
        OfMatchesWithSubs {
            pending: vec![Pending {
                slot: Slot::Iterator(iterator),
                term: Vec::new(),
                start: 0,
                end: 0,
            }],
            terms,
        }
    }

    /// `pendingQueue.add(...)`: after the equal ones already there.
    fn add(&mut self, p: Pending) {
        let at = self.pending.partition_point(|q| {
            (q.start, q.end, q.term.as_slice()) <= (p.start, p.end, p.term.as_slice())
        });
        self.pending.insert(at, p);
    }

    fn add_iterator(&mut self, it: BoxMatchesIterator) {
        let (start, end) = (it.start_offset(), it.end_offset());
        let term = self.terms.term_of(it.query());
        self.add(Pending {
            slot: Slot::Iterator(it),
            term,
            start,
            end,
        });
    }

    /// `enqueueCachedMatches(thisMI)`.
    fn enqueue_cached(&mut self, this_mi: Option<BoxMatchesIterator>) -> Result<bool> {
        let Some(mut mi) = this_mi else {
            return Ok(false);
        };
        while mi.next()? {
            let subs = mi.sub_matches()?;
            if !self.enqueue_cached(subs)? {
                let term = self.terms.term_of(mi.query());
                self.add(Pending {
                    slot: Slot::Cached,
                    term,
                    start: mi.start_offset(),
                    end: mi.end_offset(),
                });
            }
        }
        Ok(true)
    }

    /// `nextWhenMatchesIterator(miOE)` for an iterator already off the queue
    /// and positioned on a match.
    fn next_when_iterator(&mut self, mut it: BoxMatchesIterator) -> Result<()> {
        match it.sub_matches()? {
            Some(subs) => {
                self.enqueue_cached(Some(subs))?;
                if it.next()? {
                    self.add_iterator(it);
                }
            }
            // no sub-matches: it stays enqueued
            None => self.add_iterator(it),
        }
        Ok(())
    }
}

impl OffsetsEnum for OfMatchesWithSubs {
    fn next_position(&mut self) -> Result<bool> {
        if self.pending.is_empty() {
            return Ok(false);
        }
        let former = self.pending.remove(0);
        match former.slot {
            Slot::Cached => {
                if matches!(
                    self.pending.first(),
                    Some(Pending {
                        slot: Slot::Iterator(_),
                        ..
                    })
                ) {
                    let head = self.pending.remove(0);
                    if let Slot::Iterator(it) = head.slot {
                        self.next_when_iterator(it)?;
                    }
                } // else the new head is cached or there is none
            }
            Slot::Iterator(mut it) => {
                if it.next()? {
                    self.next_when_iterator(it)?;
                }
            }
        }
        Ok(!self.pending.is_empty())
    }
    fn freq(&self) -> i32 {
        // documented short-coming of the MatchesIterator-based highlighter
        1
    }
    fn term(&self) -> Option<&[u8]> {
        Some(&self.pending[0].term)
    }
    fn start_offset(&self) -> i32 {
        self.pending[0].start
    }
    fn end_offset(&self) -> i32 {
        self.pending[0].end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{PrefixQuery, TermQuery};

    /// The remembered term is the one [`query_to_term`] gives, remembered
    /// or not, for a term (an empty one too) and past the cap.
    #[test]
    fn query_terms_name_queries_as_query_to_term_does() {
        let terms = QueryTerms::default();
        let mut queries = vec![
            Clause::from(TermQuery::new("f", "t")),
            Clause::from(TermQuery::new("f", "")),
        ];
        queries.extend(
            (0..REMEMBERED_TERMS + 6).map(|i| Clause::from(PrefixQuery::new("f", format!("p{i}")))),
        );
        for _ in 0..2 {
            for q in &queries {
                assert_eq!(terms.term_of(q), query_to_term(q), "{q:?}");
            }
        }
        assert_eq!(terms.0.lock().unwrap().len(), REMEMBERED_TERMS);
    }
}
