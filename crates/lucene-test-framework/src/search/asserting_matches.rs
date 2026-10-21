//! `AssertingMatches` and `AssertingMatchesIterator`: a field's iterator
//! is only read between a `next()` that returned `true` and the one that
//! returns `false`, and never advanced once exhausted.
//!
//! Java's `AssertingWeight.matches` wraps every `Matches`; the port has no
//! `Weight`, so [`crate::search::matches_test_base::MatchesTestBase`] wraps
//! the `Matches` it reads instead.

use lucene_search::matches::{
    BoxMatches, BoxMatchesIterator, Matches, MatchesIterator, NamedMatches,
};
use lucene_search::query::Clause;
use lucene_search::Result;

/// `AssertingMatches`: each field's iterator is an
/// [`AssertingMatchesIterator`].
pub struct AssertingMatches {
    inner: BoxMatches,
}

impl AssertingMatches {
    /// Wraps `inner`.
    #[must_use]
    pub fn new(inner: BoxMatches) -> Self {
        Self { inner }
    }
}

impl Matches for AssertingMatches {
    fn get_matches(&self, field: &str) -> Result<Option<BoxMatchesIterator>> {
        Ok(self
            .inner
            .get_matches(field)?
            .map(|it| Box::new(AssertingMatchesIterator::new(it)) as BoxMatchesIterator))
    }

    fn fields(&self) -> Vec<String> {
        self.inner.fields()
    }

    fn sub_matches(&self) -> Vec<&dyn Matches> {
        self.inner.sub_matches()
    }

    fn as_named(&self) -> Option<&NamedMatches> {
        self.inner.as_named()
    }

    fn is_match_with_no_terms(&self) -> bool {
        self.inner.is_match_with_no_terms()
    }
}

/// The iterator's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Unpositioned,
    Iterating,
    Exhausted,
}

/// `AssertingMatchesIterator`.
pub struct AssertingMatchesIterator {
    inner: BoxMatchesIterator,
    state: State,
}

impl AssertingMatchesIterator {
    /// Wraps `inner`, not yet positioned.
    #[must_use]
    pub fn new(inner: BoxMatchesIterator) -> Self {
        Self {
            inner,
            state: State::Unpositioned,
        }
    }

    fn check_iterating(&self) {
        assert_eq!(
            self.state,
            State::Iterating,
            "a match read while not positioned on one"
        );
    }
}

impl MatchesIterator for AssertingMatchesIterator {
    fn next(&mut self) -> Result<bool> {
        assert_ne!(self.state, State::Exhausted, "next() after the last match");
        let more = self.inner.next()?;
        self.state = if more {
            State::Iterating
        } else {
            State::Exhausted
        };
        Ok(more)
    }

    fn start_position(&self) -> i32 {
        self.check_iterating();
        self.inner.start_position()
    }

    fn end_position(&self) -> i32 {
        self.check_iterating();
        self.inner.end_position()
    }

    fn start_offset(&self) -> i32 {
        self.check_iterating();
        self.inner.start_offset()
    }

    fn end_offset(&self) -> i32 {
        self.check_iterating();
        self.inner.end_offset()
    }

    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        self.check_iterating();
        self.inner.sub_matches()
    }

    fn query(&self) -> &Clause {
        self.check_iterating();
        self.inner.query()
    }
}
