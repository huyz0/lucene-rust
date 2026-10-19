//! `TermAutomatonQuery`, `TermAutomatonScorer` and
//! `TokenStreamToTermAutomatonQuery`: a phrase over an automaton whose
//! labels are whole terms, with an "any term" transition.
//!
//! Terms get ids in the order they are first added (`getTermID`), the any
//! term too. `finish()` widens every any transition to every id, then
//! determinizes and prunes the automaton. Searched, it first rewrites as
//! Java's `rewrite`: an empty automaton matches nothing, a one-term language
//! is a `TermQuery`, a chain of single-destination steps a `PhraseQuery` or
//! `MultiPhraseQuery`. Anything else is scored as `TermAutomatonScorer`
//! does: per document, every term's positions in position order run through
//! the automaton (`countMatches`), the frequency being the number of
//! accepted paths, and the score `simScorer.score(freq, norm)` with the
//! statistics of every query term the index holds.
//!
//! Differences: matches are resolved a segment at a time up front (the
//! scorer tree reads them as a list), where Java walks them lazily;
//! `toString`, `toDot`, `explain`, `equals`/`hashCode` (identity in Java,
//! pointer identity of the finished automaton here) and `visit` are not
//! ported; the query terms' statistics are combined in term-id order, where
//! Java iterates an `IntObjectHashMap` (the same score under BM25, whose idf
//! sums them in `double`); `TokenStreamToTermAutomatonQuery.toQuery` takes
//! the stream's tokens as [`GraphToken`]s (term, position increment and
//! length), this crate having no `TokenStream` (`lucene-analysis` sits
//! beside it, not below).

use std::collections::HashMap;
use std::sync::Arc;

use lucene_util::automaton::{operations, Automaton, Builder, RunAutomaton};

use crate::exec::extended::{norms_cursor, sim_scorer, term_entry};
use crate::exec::leaf::DocList;
use crate::exec::span::LeafPositions;
use crate::exec::{build as exec_build, BoxScorer, LeafContext, Mode, NO_MORE_DOCS};
use crate::query::{Clause, MatchNoDocsQuery, MultiPhraseQuery, PhraseQuery, TermQuery};
use crate::{Error, Result};

/// `TermAutomatonQuery`: build it state by state, then [`Self::finish`].
#[derive(Clone)]
pub struct TermAutomatonQuery {
    field: String,
    num_states: i32,
    accept: Vec<bool>,
    /// `(source, dest, termID)`, as added.
    transitions: Vec<(i32, i32, i32)>,
    term_to_id: HashMap<Option<Vec<u8>>, i32>,
    /// `idToTerm`: `None` is the any term.
    id_to_term: Vec<Option<Vec<u8>>>,
    any_term_id: i32,
    det: Option<Arc<Automaton>>,
}

impl std::fmt::Debug for TermAutomatonQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `toString`.
        write!(f, "TermAutomatonQuery(field={}", self.field)?;
        if let Some(det) = &self.det {
            write!(f, " numStates={}", det.get_num_states())?;
        }
        f.write_str(")")
    }
}

impl PartialEq for TermAutomatonQuery {
    /// Java compares by identity; a clone shares the finished automaton.
    fn eq(&self, o: &Self) -> bool {
        match (&self.det, &o.det) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl TermAutomatonQuery {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            num_states: 0,
            accept: Vec::new(),
            transitions: Vec::new(),
            term_to_id: HashMap::new(),
            id_to_term: Vec::new(),
            any_term_id: -1,
            det: None,
        }
    }

    pub fn field(&self) -> &str {
        &self.field
    }

    /// `createState()`: the new state's number.
    pub fn create_state(&mut self) -> i32 {
        self.accept.push(false);
        self.num_states = self.num_states.saturating_add(1);
        self.num_states.saturating_sub(1)
    }

    /// `setAccept(state, accept)`.
    ///
    /// # Errors
    /// A state never created (Java's `Builder` fails the same way).
    pub fn set_accept(&mut self, state: i32, accept: bool) -> Result<()> {
        let slot = usize::try_from(state)
            .ok()
            .and_then(|s| self.accept.get_mut(s))
            .ok_or_else(|| Error::IllegalArgument(format!("state={state} is out of bounds")))?;
        *slot = accept;
        Ok(())
    }

    /// `addTransition(source, dest, term)`.
    pub fn add_transition(&mut self, source: i32, dest: i32, term: impl Into<Vec<u8>>) {
        let id = self.term_id(Some(term.into()));
        self.transitions.push((source, dest, id));
    }

    /// `addAnyTransition(source, dest)`: any term at that position.
    pub fn add_any_transition(&mut self, source: i32, dest: i32) {
        let id = self.term_id(None);
        self.transitions.push((source, dest, id));
    }

    /// `getTermID`: ids in first-added order.
    fn term_id(&mut self, term: Option<Vec<u8>>) -> i32 {
        if let Some(&id) = self.term_to_id.get(&term) {
            return id;
        }
        let id = i32::try_from(self.id_to_term.len()).unwrap_or(i32::MAX);
        if term.is_none() {
            self.any_term_id = id;
        }
        self.term_to_id.insert(term.clone(), id);
        self.id_to_term.push(term);
        id
    }

    /// `finish()`, with `Operations.DEFAULT_DETERMINIZE_WORK_LIMIT`.
    ///
    /// # Errors
    /// Java's `IllegalStateException`s: an any transition leaving the
    /// initial state or entering an accept state, an automaton accepting the
    /// empty string; a transition naming a state never created; an
    /// automaton too complex to determinize.
    pub fn finish(&mut self) -> Result<()> {
        self.finish_with(operations::DEFAULT_DETERMINIZE_WORK_LIMIT)
    }

    /// `finish(determinizeWorkLimit)`.
    ///
    /// # Errors
    /// As [`Self::finish`].
    pub fn finish_with(&mut self, determinize_work_limit: i32) -> Result<()> {
        let mut b = Builder::new();
        for &accept in &self.accept {
            let s = b.create_state();
            b.set_accept(s, accept);
        }
        for &(s, d, id) in &self.transitions {
            if s < 0 || d < 0 || s >= self.num_states || d >= self.num_states {
                return Err(Error::IllegalArgument(format!(
                    "transition {s} -> {d} names a state that was never created"
                )));
            }
            b.add_transition_label(s, d, id);
        }
        let mut automaton = b.finish();
        let any = self.any_term_id;
        if any != -1 {
            let sorted = automaton.get_sorted_transitions();
            if let Some(first) = sorted.first() {
                if first.iter().any(|t| t.min <= any && any <= t.max) {
                    return Err(Error::IllegalState(
                        "automaton cannot lead with an ANY transition".into(),
                    ));
                }
            }
            for state in &sorted {
                for t in state {
                    if automaton.is_accept(t.dest) && t.min <= any && any <= t.max {
                        return Err(Error::IllegalState(
                            "automaton cannot end with an ANY transition".into(),
                        ));
                    }
                }
            }
            let term_count = i32::try_from(self.term_to_id.len()).unwrap_or(i32::MAX);
            let mut widened = Automaton::new();
            for i in 0..automaton.get_num_states() {
                widened.create_state();
                widened.set_accept(i, automaton.is_accept(i));
            }
            for state in &sorted {
                for t in state {
                    let (min, max) = if t.min <= any && any <= t.max {
                        (0, term_count.saturating_sub(1))
                    } else {
                        (t.min, t.max)
                    };
                    widened.add_transition(t.source, t.dest, min, max);
                }
            }
            widened.finish_state();
            automaton = widened;
        }
        let det = operations::determinize(&automaton, determinize_work_limit)
            .map_err(|e| Error::IllegalState(e.to_string()))?;
        let det = operations::remove_dead_states(&det);
        if det.is_accept(0) {
            // Java's `det.isAccept(0)`: an automaton left with no state (an
            // empty language) reads `false` there, finishes, and rewrites
            // to `MatchNoDocsQuery`.
            return Err(Error::IllegalState("cannot accept the empty string".into()));
        }
        self.det = Some(Arc::new(det));
        Ok(())
    }

    fn det(&self) -> Result<&Arc<Automaton>> {
        self.det
            .as_ref()
            .ok_or_else(|| Error::IllegalState(format!("Call finish first on: {self:?}")))
    }

    /// The query's terms (not the any term), for the statistics pass.
    pub(crate) fn terms(&self) -> impl Iterator<Item = &[u8]> {
        self.id_to_term.iter().filter_map(|t| t.as_deref())
    }

    fn term(&self, id: i32) -> Option<&[u8]> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.id_to_term.get(i))
            .and_then(|t| t.as_deref())
    }

    /// `rewrite(searcher)`: `None` when the query rewrites to itself.
    ///
    /// # Errors
    /// An unfinished query.
    pub fn rewrite(&self) -> Result<Option<Clause>> {
        let det = self.det()?;
        if operations::is_empty(det) {
            return Ok(Some(Clause::MatchNoDocs(MatchNoDocsQuery::new())));
        }
        if let Ok(Some(single)) = operations::get_singleton(det) {
            if let [id] = single.as_slice() {
                if let Some(t) = self.term(*id) {
                    return Ok(Some(Clause::Term(TermQuery::new(self.field.clone(), t))));
                }
            }
        }
        let any = self.any_term_id;
        let sorted = det.get_sorted_transitions();
        // `(terms, position)` per step; `phrase` false once a step has more
        // than one term.
        let mut steps: Vec<(Vec<Vec<u8>>, i32)> = Vec::new();
        let mut phrase = true;
        let (mut state, mut pos) = (0i32, 0i32);
        loop {
            let ts = usize::try_from(state)
                .ok()
                .and_then(|s| sorted.get(s))
                .map_or(&[][..], Vec::as_slice);
            if ts.is_empty() {
                if !det.is_accept(state) {
                    return Ok(None);
                }
                break;
            } else if det.is_accept(state) {
                return Ok(None);
            }
            let dest = ts[0].dest;
            let mut terms = Vec::new();
            let mut matches_any = false;
            for t in ts {
                if t.dest != dest {
                    return Ok(None);
                }
                matches_any |= any >= t.min && any <= t.max;
                if !matches_any {
                    for id in t.min..=t.max {
                        terms.push(self.term(id).unwrap_or_default().to_vec());
                    }
                }
            }
            if !matches_any {
                phrase &= terms.len() == 1;
                steps.push((terms, pos));
            }
            state = dest;
            pos = pos.saturating_add(1);
        }
        Ok(Some(if phrase {
            let mut pq =
                PhraseQuery::new(self.field.clone(), steps.iter().map(|(t, _)| t[0].clone()));
            pq.positions = steps.iter().map(|&(_, p)| p).collect();
            Clause::Phrase(pq)
        } else {
            Clause::MultiPhrase(MultiPhraseQuery::with_positions(self.field.clone(), steps)?)
        }))
    }
}

/// `TokenStreamToTermAutomatonQuery`: a token graph's paths as a
/// [`TermAutomatonQuery`], the term `*` standing for any term.
#[derive(Debug, Clone)]
pub struct TokenStreamToTermAutomatonQuery {
    preserve_position_increments: bool,
}

impl Default for TokenStreamToTermAutomatonQuery {
    fn default() -> Self {
        Self {
            preserve_position_increments: true,
        }
    }
}

/// One token: its bytes, position increment and position length.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphToken {
    pub term: Vec<u8>,
    pub position_increment: i32,
    pub position_length: i32,
}

impl GraphToken {
    pub fn new(term: impl Into<Vec<u8>>, position_increment: i32, position_length: i32) -> Self {
        Self {
            term: term.into(),
            position_increment,
            position_length,
        }
    }
}

impl TokenStreamToTermAutomatonQuery {
    pub fn new() -> Self {
        Self::default()
    }

    /// `setPreservePositionIncrements`: when false, a gap counts as one.
    pub fn set_preserve_position_increments(&mut self, enable: bool) {
        self.preserve_position_increments = enable;
    }

    /// `toQuery(field, in)` over the stream's tokens.
    ///
    /// # Errors
    /// `IllegalArgumentException("cannot handle holes; ...")` for a position
    /// increment above one; what [`TermAutomatonQuery::finish`] reports.
    pub fn to_query(
        &self,
        field: impl Into<String>,
        tokens: impl IntoIterator<Item = GraphToken>,
    ) -> Result<TermAutomatonQuery> {
        let mut query = TermAutomatonQuery::new(field);
        let (mut pos, mut state) = (-1i32, -1i32);
        for token in tokens {
            let mut inc = token.position_increment;
            if !self.preserve_position_increments && inc > 1 {
                inc = 1;
            }
            if inc > 1 {
                return Err(Error::IllegalArgument(
                    "cannot handle holes; to accept any term, use '*' term".into(),
                ));
            }
            if inc > 0 {
                pos = pos.saturating_add(inc);
            }
            let end = pos.saturating_add(token.position_length);
            while state < end {
                state = query.create_state();
            }
            if token.term.as_slice() == b"*" {
                query.add_any_transition(pos, end);
            } else {
                query.add_transition(pos, end, token.term);
            }
        }
        query.set_accept(state, true)?;
        query.finish()?;
        Ok(query)
    }
}

// ---------------------------------------------------------------------------
// The weight and scorer
// ---------------------------------------------------------------------------

/// `TermAutomatonWeight.scorerSupplier(...).get(...)`, after `rewrite`.
pub(crate) fn scorer<'a>(
    ctx: &LeafContext<'a>,
    q: &TermAutomatonQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    if let Some(rewritten) = q.rewrite()? {
        return exec_build::build(ctx, &rewritten, boost, mode, top_level);
    }
    let det = q.det()?;
    // `allTermStats`: every query term the index holds.
    let mut stats = Vec::new();
    let mut collection = None;
    for term in q.terms() {
        if let Some(entry) = term_entry(ctx, &q.field, term)? {
            if entry.doc_freq > 0 {
                collection.get_or_insert(entry.collection_statistics());
                stats.push(entry.term_statistics());
            }
        }
    }
    let Some(collection) = collection else {
        return Ok(None);
    };
    let Some(pos_in) = ctx.pos_in else {
        return Err(Error::MissingPosInput);
    };
    let mut subs: Vec<(i32, LeafPositions<'a>)> = Vec::new();
    for (id, term) in q.id_to_term.iter().enumerate() {
        if let Some(term) = term {
            let leaf = LeafPositions::open(ctx, pos_in, &q.field, term)?;
            if !matches!(leaf, LeafPositions::Absent) {
                subs.push((i32::try_from(id).unwrap_or(i32::MAX), leaf));
            }
        }
    }
    if subs.is_empty() {
        return Ok(None);
    }
    let alphabet = i32::try_from(q.id_to_term.len()).unwrap_or(i32::MAX);
    let run = RunAutomaton::new((**det).clone(), alphabet)
        .map_err(|e| Error::IllegalState(e.to_string()))?;
    let sim = sim_scorer(ctx, &q.field, boost, &collection, &stats);
    let mut norms = norms_cursor(ctx, &q.field);
    let mut counter = MatchCounter::new(&run, q.any_term_id);
    let (mut docs, mut scores) = (Vec::new(), Vec::new());
    let mut events: Vec<(i32, i32)> = Vec::new();
    let mut scratch = Vec::new();
    let mut doc = -1i32;
    loop {
        // The union's next document (`docIDQueue`).
        let mut next = NO_MORE_DOCS;
        for (_, leaf) in subs.iter_mut() {
            next = next.min(leaf.advance(doc.saturating_add(1))?);
        }
        doc = next;
        if doc == NO_MORE_DOCS {
            break;
        }
        if !ctx.live_docs.is_none_or(|l| l.get_doc(doc)) {
            continue;
        }
        events.clear();
        for (id, leaf) in subs.iter_mut() {
            scratch.clear();
            leaf.positions_at(doc, &mut scratch)?;
            events.extend(scratch.iter().map(|&p| (p, *id)));
        }
        // `posQueue`: position order; terms on one position step from the
        // same states, in any order, to the same counts.
        events.sort_by_key(|&(p, _)| p);
        let freq = counter.count(&events);
        if freq > 0 {
            docs.push(doc);
            if mode.needs_scores() {
                let norm = match norms.as_mut() {
                    Some(n) => n.norm_long(doc)?.unwrap_or(1),
                    None => 1,
                };
                scores.push(sim.score(freq as f32, norm));
            }
        }
    }
    if docs.is_empty() {
        return Ok(None);
    }
    Ok(Some(Box::new(DocList::new(docs, scores))))
}

/// `TermAutomatonScorer.countMatches`' state: the automaton states alive at
/// each position from `pos_shift` on (`PosState`s).
struct MatchCounter<'r> {
    run: &'r RunAutomaton,
    any: i32,
    positions: Vec<Vec<i32>>,
    pos_shift: i32,
}

impl<'r> MatchCounter<'r> {
    fn new(run: &'r RunAutomaton, any: i32) -> Self {
        Self {
            run,
            any,
            positions: vec![Vec::new(); 4],
            pos_shift: 0,
        }
    }

    /// `getPosition(pos)`'s slot.
    fn slot(&self, pos: i32) -> usize {
        usize::try_from(pos.saturating_sub(self.pos_shift)).unwrap_or(0)
    }

    /// `shift(pos)`.
    fn shift(&mut self, pos: i32) {
        let limit = self.slot(pos);
        for p in self.positions.iter_mut().take(limit) {
            p.clear();
        }
        self.pos_shift = pos;
    }

    /// Steps every state alive at `from` by `label` into `to`, counting the
    /// accepted ones when `count` is set.
    fn step_all(&mut self, from: usize, to: usize, label: i32, count: bool) -> u32 {
        let mut accepted = 0u32;
        for i in 0..self.positions[from].len() {
            let state = self.run.step(self.positions[from][i], label);
            if state != -1 {
                self.positions[to].push(state);
                if count && self.run.is_accept(state) {
                    accepted = accepted.saturating_add(1);
                }
            }
        }
        accepted
    }

    /// The number of accepted paths over `events`, `(position, termID)` in
    /// position order.
    fn count(&mut self, events: &[(i32, i32)]) -> u32 {
        let mut freq = 0u32;
        let mut last_pos = -1i32;
        self.pos_shift = -1;
        for &(pos, term) in events {
            if self.pos_shift == -1 {
                self.pos_shift = pos;
            }
            let need = self.slot(pos).saturating_add(2);
            if need > self.positions.len() {
                self.positions.resize(need.next_power_of_two(), Vec::new());
            }
            if last_pos != -1 && self.any != -1 {
                // Positions no term filled: every live state steps by the
                // any term until one has none.
                let start = last_pos;
                while last_pos < pos {
                    let at = self.slot(last_pos);
                    if self.positions[at].is_empty() && last_pos > start {
                        // Petered out (Java's `lastPos = pos` here is
                        // overwritten before it is read).
                        break;
                    }
                    let next = self.slot(last_pos.saturating_add(1));
                    self.step_all(at, next, self.any, false);
                    last_pos = last_pos.saturating_add(1);
                }
            }
            let (mut at, mut next) = (self.slot(pos), self.slot(pos.saturating_add(1)));
            if self.positions[at].is_empty() && self.positions[next].is_empty() {
                self.shift(pos);
                at = self.slot(pos);
                next = self.slot(pos.saturating_add(1));
            }
            freq = freq.saturating_add(self.step_all(at, next, term, true));
            let state = self.run.step(0, term);
            if state != -1 {
                self.positions[next].push(state);
                if self.run.is_accept(state) {
                    freq = freq.saturating_add(1);
                }
            }
            last_pos = pos;
        }
        let limit = self.slot(last_pos.saturating_add(1));
        for p in self.positions.iter_mut().take(limit.saturating_add(1)) {
            p.clear();
        }
        freq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finished(
        states: i32,
        accept: &[i32],
        edges: &[(i32, i32, &str)],
    ) -> Result<TermAutomatonQuery> {
        let mut q = TermAutomatonQuery::new("f");
        for _ in 0..states {
            q.create_state();
        }
        for &a in accept {
            q.set_accept(a, true)?;
        }
        for &(s, d, t) in edges {
            if t == "*" {
                q.add_any_transition(s, d);
            } else {
                q.add_transition(s, d, t);
            }
        }
        q.finish()?;
        Ok(q)
    }

    #[test]
    fn finish_rejects_what_java_rejects() {
        let msg = |r: Result<TermAutomatonQuery>| r.err().map(|e| e.to_string());
        assert_eq!(
            msg(finished(3, &[2], &[(0, 1, "*"), (1, 2, "a")])).as_deref(),
            Some("automaton cannot lead with an ANY transition")
        );
        assert_eq!(
            msg(finished(3, &[2], &[(0, 1, "a"), (1, 2, "*")])).as_deref(),
            Some("automaton cannot end with an ANY transition")
        );
        assert_eq!(
            msg(finished(2, &[0, 1], &[(0, 1, "a")])).as_deref(),
            Some("cannot accept the empty string")
        );
        assert!(msg(finished(2, &[1], &[(0, 5, "a")])).is_some());
        // An empty language finishes (no state is left to accept) and
        // rewrites to nothing.
        let empty = finished(1, &[], &[]).unwrap();
        assert_eq!(
            empty.rewrite().unwrap(),
            Some(Clause::MatchNoDocs(MatchNoDocsQuery::new()))
        );
        let mut q = TermAutomatonQuery::new("f");
        assert!(q.set_accept(0, true).is_err());
        assert!(q.rewrite().is_err());
        assert_eq!(format!("{q:?}"), "TermAutomatonQuery(field=f)");
        assert_ne!(q, q.clone());
    }

    #[test]
    fn rewrites_as_java_does() {
        let q = finished(2, &[1], &[(0, 1, "a")]).unwrap();
        assert_eq!(
            q.rewrite().unwrap(),
            Some(Clause::Term(TermQuery::new("f", "a")))
        );
        assert_eq!(format!("{q:?}"), "TermAutomatonQuery(field=f numStates=2)");
        assert_eq!(q, q.clone());
        let q = finished(4, &[3], &[(0, 1, "a"), (1, 2, "*"), (2, 3, "b")]).unwrap();
        let mut pq = PhraseQuery::new("f", ["a", "b"]);
        pq.positions = vec![0, 2];
        assert_eq!(q.rewrite().unwrap(), Some(Clause::Phrase(pq)));
        let q = finished(3, &[2], &[(0, 1, "a"), (0, 1, "c"), (1, 2, "b")]).unwrap();
        let Some(Clause::MultiPhrase(m)) = q.rewrite().unwrap() else {
            panic!("multi-phrase");
        };
        assert_eq!(m.term_arrays.len(), 2);
        // An accept state with transitions out, two destinations, and an
        // empty language stay the query itself or match nothing.
        let q = finished(3, &[1, 2], &[(0, 1, "a"), (1, 2, "a")]).unwrap();
        assert_eq!(q.rewrite().unwrap(), None);
        let q = finished(3, &[1, 2], &[(0, 1, "a"), (0, 2, "b")]).unwrap();
        assert_eq!(q.rewrite().unwrap(), None);
        let q = finished(3, &[2], &[(0, 1, "a"), (1, 1, "*"), (1, 2, "b")]).unwrap();
        assert_eq!(q.rewrite().unwrap(), None);
    }

    #[test]
    fn token_streams_become_automata() {
        let ts = TokenStreamToTermAutomatonQuery::new();
        let q = ts
            .to_query(
                "f",
                [GraphToken::new("a", 1, 1), GraphToken::new("b", 1, 1)],
            )
            .unwrap();
        let mut pq = PhraseQuery::new("f", ["a", "b"]);
        pq.positions = vec![0, 1];
        assert_eq!(q.rewrite().unwrap(), Some(Clause::Phrase(pq)));
        let hole = [GraphToken::new("a", 1, 1), GraphToken::new("b", 2, 1)];
        assert_eq!(
            ts.to_query("f", hole.clone()).unwrap_err().to_string(),
            "cannot handle holes; to accept any term, use '*' term"
        );
        let mut flat = TokenStreamToTermAutomatonQuery::new();
        flat.set_preserve_position_increments(false);
        assert!(flat.to_query("f", hole).is_ok());
        assert_eq!(GraphToken::new("x", 1, 1).term, b"x");
    }

    #[test]
    fn counts_accepted_paths_through_any_terms() {
        // a * b: ids a=0, any=1, b=2.
        let q = finished(4, &[3], &[(0, 1, "a"), (1, 2, "*"), (2, 3, "b")]).unwrap();
        let run = RunAutomaton::new((**q.det.as_ref().unwrap()).clone(), 3).unwrap();
        let mut c = MatchCounter::new(&run, q.any_term_id);
        assert_eq!(c.count(&[(0, 0), (2, 2)]), 1);
        assert_eq!(c.count(&[(0, 0), (1, 2)]), 0);
        assert_eq!(c.count(&[(0, 0), (3, 2)]), 0);
        // Far-apart positions shift the window and grow it.
        assert_eq!(c.count(&[(5, 0), (40, 0), (42, 2), (100, 0), (102, 2)]), 2);
    }
}
