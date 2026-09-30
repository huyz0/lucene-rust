//! Port of `org.apache.lucene.index.FilteredTermsEnum`: a [`TermsEnum`] that
//! walks another one and keeps the terms an `accept()` says to, seeking ahead
//! whenever `accept()` asks for it -- the extension point `MultiTermQuery`s
//! build their term enumerations on.
//!
//! Java subclasses `FilteredTermsEnum` and overrides `accept`/`nextSeekTerm`;
//! here the two hooks are a [`TermFilter`] the enum owns. Seeking the filtered
//! enum itself is refused (`UnsupportedOperationException`), as in Java.
//!
//! [`AutomatonTermsEnum`] is the one filter Java itself ships in this package:
//! the terms a DFA accepts, the default `Terms.intersect`.

use lucene_codecs::automaton::{ByteDfa, DfaWalker, Verdict};
use lucene_codecs::blocktree::SeekStatus;

use super::{ImpactsEnum, PostingsEnum, PostingsFlags, TermsEnum};
use crate::{Error, Result};

/// `FilteredTermsEnum.AcceptStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptStatus {
    /// Accept the term and move to the next one.
    Yes,
    /// Accept the term, then seek to [`TermFilter::next_seek_term`].
    YesAndSeek,
    /// Reject the term and move to the next one.
    No,
    /// Reject the term, then seek to [`TermFilter::next_seek_term`].
    NoAndSeek,
    /// Reject the term and stop.
    End,
}

/// `FilteredTermsEnum`'s two overridable methods.
pub trait TermFilter {
    /// `accept(term)`.
    fn accept(&mut self, term: &[u8]) -> Result<AcceptStatus>;
    /// `nextSeekTerm(currentTerm)`: where to seek next, `None` to stop.
    /// `initial` is the term `setInitialSeekTerm` recorded, handed over once;
    /// Java's default returns it, and so does this one.
    fn next_seek_term(
        &mut self,
        _current: Option<&[u8]>,
        initial: Option<Vec<u8>>,
    ) -> Result<Option<Vec<u8>>> {
        Ok(initial)
    }
}

/// `FilteredTermsEnum`.
pub struct FilteredTermsEnum<'a, F: TermFilter> {
    tenum: Box<dyn TermsEnum + 'a>,
    filter: F,
    initial_seek_term: Option<Vec<u8>>,
    do_seek: bool,
    /// `actualTerm`: the last term the underlying enum stood on.
    actual: Option<Vec<u8>>,
}

impl<'a, F: TermFilter> FilteredTermsEnum<'a, F> {
    /// `FilteredTermsEnum(tenum, startWithSeek)`. With `start_with_seek` the
    /// first `next()` seeks to [`TermFilter::next_seek_term`] -- which, with
    /// no initial seek term, ends the enumeration at once, exactly as in Java.
    pub fn new(tenum: Box<dyn TermsEnum + 'a>, filter: F, start_with_seek: bool) -> Self {
        Self {
            tenum,
            filter,
            initial_seek_term: None,
            do_seek: start_with_seek,
            actual: None,
        }
    }

    /// `setInitialSeekTerm(term)`.
    pub fn set_initial_seek_term(&mut self, term: Option<Vec<u8>>) {
        self.initial_seek_term = term;
    }

    /// The filter.
    pub fn filter(&self) -> &F {
        &self.filter
    }

    fn unsupported() -> Error {
        Error::Unsupported("FilteredTermsEnum does not support seeking".into())
    }
}

impl<F: TermFilter> TermsEnum for FilteredTermsEnum<'_, F> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        loop {
            if self.do_seek {
                self.do_seek = false;
                let initial = self.initial_seek_term.take();
                let t = self
                    .filter
                    .next_seek_term(self.actual.as_deref(), initial)?;
                let Some(t) = t else {
                    return Ok(None);
                };
                if self.tenum.try_seek_ceil(&t)? == SeekStatus::End {
                    return Ok(None);
                }
                self.actual = self.tenum.term().map(<[u8]>::to_vec);
            } else {
                self.actual = self.tenum.next()?.map(<[u8]>::to_vec);
            }
            let Some(term) = self.actual.as_deref() else {
                return Ok(None);
            };
            match self.filter.accept(term)? {
                AcceptStatus::YesAndSeek => {
                    self.do_seek = true;
                    return Ok(self.tenum.term());
                }
                AcceptStatus::Yes => return Ok(self.tenum.term()),
                AcceptStatus::NoAndSeek => self.do_seek = true,
                AcceptStatus::End => return Ok(None),
                AcceptStatus::No => {}
            }
        }
    }

    fn term(&self) -> Option<&[u8]> {
        self.tenum.term()
    }

    fn try_seek_ceil(&mut self, _target: &[u8]) -> Result<SeekStatus> {
        Err(Self::unsupported())
    }

    fn try_seek_exact(&mut self, _target: &[u8]) -> Result<bool> {
        Err(Self::unsupported())
    }

    fn seek_exact_ord(&mut self, _ord: i64) -> Result<()> {
        Err(Self::unsupported())
    }

    fn doc_freq(&mut self) -> Result<i32> {
        self.tenum.doc_freq()
    }

    fn total_term_freq(&mut self) -> Result<i64> {
        self.tenum.total_term_freq()
    }

    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        self.tenum.postings(flags)
    }

    fn impacts(&mut self, flags: PostingsFlags) -> Result<Box<dyn ImpactsEnum>> {
        self.tenum.impacts(flags)
    }

    fn ord(&self) -> Result<i64> {
        self.tenum.ord()
    }
}

/// The [`TermFilter`] of `AutomatonTermsEnum`: the terms `dfa` accepts.
/// Where a term's prefix is dead it seeks to the next string the automaton
/// can still extend ([`DfaWalker::next_live_after`], `nextString`), so a
/// walk skips every dead branch rather than every term in it.
pub struct AutomatonFilter<'d> {
    dfa: &'d ByteDfa,
    walker: DfaWalker,
    pending_seek: Option<Vec<u8>>,
    /// `intersect(compiled, startTerm)`: the start term itself is excluded.
    exclude: Option<Vec<u8>>,
}

impl TermFilter for AutomatonFilter<'_> {
    fn accept(&mut self, term: &[u8]) -> Result<AcceptStatus> {
        if self.exclude.as_deref() == Some(term) {
            return Ok(AcceptStatus::No);
        }
        Ok(match self.walker.feed(self.dfa, term) {
            Verdict::Accept => AcceptStatus::Yes,
            Verdict::Reject => AcceptStatus::No,
            Verdict::DeadAt(k) => match self.walker.next_live_after(self.dfa, term, k) {
                Some(target) => {
                    self.pending_seek = Some(target);
                    AcceptStatus::NoAndSeek
                }
                None => AcceptStatus::End,
            },
        })
    }

    fn next_seek_term(
        &mut self,
        _current: Option<&[u8]>,
        initial: Option<Vec<u8>>,
    ) -> Result<Option<Vec<u8>>> {
        Ok(self.pending_seek.take().or(initial))
    }
}

/// `AutomatonTermsEnum`: a [`FilteredTermsEnum`] over the terms a DFA
/// accepts, strictly after an optional start term.
pub struct AutomatonTermsEnum<'a>(FilteredTermsEnum<'a, AutomatonFilter<'a>>);

impl<'a> AutomatonTermsEnum<'a> {
    /// The terms of `tenum` that `dfa` accepts, after `start_term` when given.
    pub fn new(
        tenum: Box<dyn TermsEnum + 'a>,
        dfa: &'a ByteDfa,
        start_term: Option<&[u8]>,
    ) -> Self {
        let filter = AutomatonFilter {
            dfa,
            walker: DfaWalker::new(dfa),
            pending_seek: None,
            exclude: start_term.map(<[u8]>::to_vec),
        };
        let mut inner = FilteredTermsEnum::new(tenum, filter, start_term.is_some());
        inner.set_initial_seek_term(start_term.map(<[u8]>::to_vec));
        Self(inner)
    }
}

impl TermsEnum for AutomatonTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        self.0.next()
    }
    fn term(&self) -> Option<&[u8]> {
        self.0.term()
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.0.try_seek_ceil(target)
    }
    fn doc_freq(&mut self) -> Result<i32> {
        self.0.doc_freq()
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        self.0.total_term_freq()
    }
    fn postings(&mut self, flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        self.0.postings(flags)
    }
}
