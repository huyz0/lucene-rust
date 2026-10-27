//! `SortedDocValues.termsEnum()`/`SortedSetDocValues.termsEnum()` and their
//! `intersect`: Java's default `SortedDocValuesTermsEnum` and
//! `SortedSetDocValuesTermsEnum`, a field's ordinal dictionary walked as a
//! [`TermsEnum`] -- positioned by `lookupTerm`, read by `lookupOrd`.
//!
//! The enum owns its values (Java shares the `SortedDocValues` object with
//! the caller; a port value reader is a `&mut` cursor, so a caller wanting
//! both opens the field twice). Postings and statistics are refused, as
//! Java's `UnsupportedOperationException`.

use lucene_codecs::automaton::ByteDfa;
use lucene_codecs::blocktree::SeekStatus;

use super::filtered_terms_enum::AutomatonTermsEnum;
use super::{PostingsEnum, PostingsFlags, SortedDocValues, SortedSetDocValues, TermsEnum};
use crate::{Error, Result};

/// The ordinal dictionary a [`DocValuesTermsEnum`] walks.
enum Dictionary<'a> {
    Sorted(Box<dyn SortedDocValues + 'a>),
    SortedSet(Box<dyn SortedSetDocValues + 'a>),
}

impl Dictionary<'_> {
    fn value_count(&self) -> i64 {
        match self {
            Dictionary::Sorted(v) => i64::from(v.value_count()),
            Dictionary::SortedSet(v) => v.value_count(),
        }
    }

    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        match self {
            Dictionary::Sorted(v) => {
                let ord = i32::try_from(ord).map_err(|_| {
                    Error::Unsupported(format!("ord {ord} is past a SORTED field's int range"))
                })?;
                v.lookup_ord(ord)
            }
            Dictionary::SortedSet(v) => v.lookup_ord(ord),
        }
    }

    // SENTINEL: negative = absent, `-insertionPoint-1` (`lookupTerm`).
    fn lookup_term(&mut self, key: &[u8]) -> Result<i64> {
        match self {
            Dictionary::Sorted(v) => v.lookup_term(key).map(i64::from),
            Dictionary::SortedSet(v) => v.lookup_term(key),
        }
    }
}

/// `SortedDocValuesTermsEnum`/`SortedSetDocValuesTermsEnum`.
pub struct DocValuesTermsEnum<'a> {
    values: Dictionary<'a>,
    /// `-1` before the first term; the value count once exhausted.
    current_ord: i64,
    /// The current term, `None` when unpositioned or exhausted.
    term: Option<Vec<u8>>,
}

impl<'a> DocValuesTermsEnum<'a> {
    /// `values.termsEnum()` of a `SORTED` field.
    pub fn sorted(values: Box<dyn SortedDocValues + 'a>) -> Self {
        Self::new(Dictionary::Sorted(values))
    }

    /// `values.termsEnum()` of a `SORTED_SET` field.
    pub fn sorted_set(values: Box<dyn SortedSetDocValues + 'a>) -> Self {
        Self::new(Dictionary::SortedSet(values))
    }

    fn new(values: Dictionary<'a>) -> Self {
        Self {
            values,
            current_ord: -1,
            term: None,
        }
    }

    /// `values.intersect(compiled)` for a `NORMAL` automaton: the terms
    /// `dfa` accepts, through [`AutomatonTermsEnum`] (Java's default).
    pub fn intersect<'d>(self, dfa: &'d ByteDfa) -> AutomatonTermsEnum<'d>
    where
        'a: 'd,
    {
        AutomatonTermsEnum::new(Box::new(self), dfa, None)
    }
}

impl TermsEnum for DocValuesTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        let count = self.values.value_count();
        if self.current_ord >= count {
            return Ok(None);
        }
        self.current_ord += 1;
        if self.current_ord >= count {
            self.term = None;
            return Ok(None);
        }
        self.term = Some(self.values.lookup_ord(self.current_ord)?);
        Ok(self.term.as_deref())
    }

    fn term(&self) -> Option<&[u8]> {
        self.term.as_deref()
    }

    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let ord = self.values.lookup_term(target)?;
        if ord >= 0 {
            self.current_ord = ord;
            self.term = Some(target.to_vec());
            return Ok(SeekStatus::Found);
        }
        self.current_ord = -ord - 1;
        if self.current_ord == self.values.value_count() {
            self.term = None;
            Ok(SeekStatus::End)
        } else {
            self.term = Some(self.values.lookup_ord(self.current_ord)?);
            Ok(SeekStatus::NotFound)
        }
    }

    fn try_seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        let ord = self.values.lookup_term(target)?;
        if ord >= 0 {
            self.current_ord = ord;
            self.term = Some(target.to_vec());
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn doc_freq(&mut self) -> Result<i32> {
        Err(unsupported("docFreq"))
    }

    fn total_term_freq(&mut self) -> Result<i64> {
        Err(unsupported("totalTermFreq"))
    }

    fn postings(&mut self, _flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        Err(unsupported("postings"))
    }

    fn ord(&self) -> Result<i64> {
        Ok(self.current_ord)
    }

    fn seek_exact_ord(&mut self, ord: i64) -> Result<()> {
        if ord < 0 || ord >= self.values.value_count() {
            return Err(Error::Unsupported(format!(
                "ord {ord} is out of bounds for {} values",
                self.values.value_count()
            )));
        }
        self.current_ord = ord;
        self.term = Some(self.values.lookup_ord(ord)?);
        Ok(())
    }

    /// An `OrdTermState` of the current ordinal.
    fn term_state(&mut self) -> Result<super::TermState> {
        Ok(super::TermState::ord(self.current_ord))
    }

    /// `seekExact(state.ord)`: the term is not looked at.
    fn seek_exact_state(&mut self, _term: &[u8], state: &super::TermState) -> Result<()> {
        let ord = state
            .as_ord()
            .ok_or_else(|| Error::IllegalArgument("not an OrdTermState".into()))?;
        self.seek_exact_ord(ord)
    }
}

fn unsupported(what: &str) -> Error {
    Error::Unsupported(format!("a doc values TermsEnum has no {what}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::{DocIdSetIterator, DocValuesIterator, NO_MORE_DOCS};

    /// A dictionary with no documents.
    struct Dict(Vec<&'static [u8]>);

    impl DocIdSetIterator for Dict {
        fn doc_id(&self) -> i32 {
            NO_MORE_DOCS
        }
        fn next_doc(&mut self) -> Result<i32> {
            Ok(NO_MORE_DOCS)
        }
        fn advance(&mut self, _target: i32) -> Result<i32> {
            Ok(NO_MORE_DOCS)
        }
        fn cost(&self) -> i64 {
            0
        }
    }

    impl DocValuesIterator for Dict {
        fn advance_exact(&mut self, _target: i32) -> Result<bool> {
            Ok(false)
        }
    }

    impl SortedDocValues for Dict {
        fn ord_value(&self) -> i32 {
            -1
        }
        fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
            Ok(self.0[ord as usize].to_vec())
        }
        fn value_count(&self) -> i32 {
            self.0.len() as i32
        }
    }

    impl SortedSetDocValues for Dict {
        fn doc_value_count(&self) -> i32 {
            0
        }
        fn next_ord(&mut self) -> Result<i64> {
            Ok(-1)
        }
        fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
            Ok(self.0[ord as usize].to_vec())
        }
        fn value_count(&self) -> i64 {
            self.0.len() as i64
        }
    }

    fn three() -> Dict {
        Dict(vec![b"beer", b"hello", b"world"])
    }

    #[test]
    fn walks_seeks_and_reads_ords() {
        for mut te in [
            DocValuesTermsEnum::sorted(Box::new(three())),
            DocValuesTermsEnum::sorted_set(Box::new(three())),
        ] {
            assert_eq!(te.term(), None);
            assert_eq!(te.ord().unwrap(), -1);
            assert_eq!(te.next().unwrap(), Some(&b"beer"[..]));
            assert_eq!(te.next().unwrap(), Some(&b"hello"[..]));
            assert_eq!(te.ord().unwrap(), 1);
            assert_eq!(te.next().unwrap(), Some(&b"world"[..]));
            assert_eq!(te.next().unwrap(), None);
            assert_eq!(te.next().unwrap(), None, "stays exhausted");
            assert_eq!(te.term(), None);

            assert_eq!(te.try_seek_ceil(b"ha!").unwrap(), SeekStatus::NotFound);
            assert_eq!((te.term(), te.ord().unwrap()), (Some(&b"hello"[..]), 1));
            assert_eq!(te.try_seek_ceil(b"beer").unwrap(), SeekStatus::Found);
            assert_eq!(te.ord().unwrap(), 0);
            assert_eq!(te.try_seek_ceil(b"zzz").unwrap(), SeekStatus::End);
            assert_eq!(te.term(), None);

            assert!(te.try_seek_exact(b"world").unwrap());
            assert_eq!(te.ord().unwrap(), 2);
            assert!(!te.try_seek_exact(b"bogus").unwrap());

            te.seek_exact_ord(1).unwrap();
            assert_eq!(te.term(), Some(&b"hello"[..]));
            assert!(te.seek_exact_ord(3).is_err());
            assert!(te.seek_exact_ord(-1).is_err());

            assert!(te.doc_freq().is_err());
            assert!(te.total_term_freq().is_err());
            assert!(te.postings(PostingsFlags::None).is_err());
        }
    }

    #[test]
    fn intersect_keeps_the_accepted_terms_with_their_ords() {
        use lucene_codecs::automaton::Nfa;
        // Accepts exactly the terms starting with 'h' or 'w'.
        let mut nfa = Nfa::new();
        let start = nfa.state().unwrap();
        let rest = nfa.state().unwrap();
        let accept = nfa.state().unwrap();
        nfa.range(start, b'h', b'h', rest);
        nfa.range(start, b'w', b'w', rest);
        nfa.range(rest, 0, 255, rest);
        nfa.epsilon(rest, accept);
        let dfa = nfa.determinize(start, accept).unwrap();
        let mut te = DocValuesTermsEnum::sorted(Box::new(three())).intersect(&dfa);
        let mut seen = Vec::new();
        while let Some(term) = te.next().unwrap() {
            let term = term.to_vec();
            seen.push((term, te.ord().unwrap()));
        }
        assert_eq!(seen, vec![(b"hello".to_vec(), 1), (b"world".to_vec(), 2)]);
    }

    #[test]
    fn a_sorted_ord_past_the_int_range_is_refused() {
        let mut te = DocValuesTermsEnum::sorted(Box::new(three()));
        te.current_ord = -1;
        assert!(te.values.lookup_ord(i64::from(i32::MAX) + 1).is_err());
    }
}
