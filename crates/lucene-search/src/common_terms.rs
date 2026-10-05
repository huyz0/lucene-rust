//! `lucene-queries`' `CommonTermsQuery` (`org.apache.lucene.queries`,
//! Lucene 10.5.0): a disjunction of terms that rewrites, against the
//! searcher's reader, into a required boolean of the rare terms and an
//! optional boolean of the frequent ones -- a term is frequent when its
//! document frequency passes `maxTermFrequency`, a count (`>= 1`) or a
//! fraction of `maxDoc`.
//!
//! The query is a [`crate::query::Clause::Extended`] leaf rewritten where
//! `IndexSearcher.rewrite` runs (the searcher's rewrite pass,
//! [`crate::rescorer::rewrite_rescore_clauses`]); its `toString` is Java's.

use crate::index_searcher::IndexSearcher;
use crate::query::{BooleanQuery, BoostQuery, Clause, MatchNoDocsQuery, TermQuery};
use crate::query_visitor::Occur;
use crate::{Error, Result};

/// `CommonTermsQuery`.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTermsQuery {
    /// `(field, term)`, in the order they were added.
    pub terms: Vec<(String, Vec<u8>)>,
    pub max_term_frequency: f32,
    pub low_freq_occur: Occur,
    pub high_freq_occur: Occur,
    pub low_freq_boost: f32,
    pub high_freq_boost: f32,
    pub low_freq_min_nr_should_match: f32,
    pub high_freq_min_nr_should_match: f32,
}

impl CommonTermsQuery {
    /// `new CommonTermsQuery(highFreqOccur, lowFreqOccur, maxTermFrequency)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a `MUST_NOT` occur, as Java's
    /// constructor throws.
    pub fn new(
        high_freq_occur: Occur,
        low_freq_occur: Occur,
        max_term_frequency: f32,
    ) -> Result<Self> {
        if high_freq_occur == Occur::MustNot {
            return Err(Error::IllegalArgument(
                "highFreqOccur should be MUST or SHOULD but was MUST_NOT".into(),
            ));
        }
        if low_freq_occur == Occur::MustNot {
            return Err(Error::IllegalArgument(
                "lowFreqOccur should be MUST or SHOULD but was MUST_NOT".into(),
            ));
        }
        Ok(CommonTermsQuery {
            terms: Vec::new(),
            max_term_frequency,
            low_freq_occur,
            high_freq_occur,
            low_freq_boost: 1.0,
            high_freq_boost: 1.0,
            low_freq_min_nr_should_match: 0.0,
            high_freq_min_nr_should_match: 0.0,
        })
    }

    /// `add(term)`.
    pub fn add(&mut self, field: impl Into<String>, term: impl Into<Vec<u8>>) {
        self.terms.push((field.into(), term.into()));
    }

    /// `minNrShouldMatch`: a count from `1` on, else a fraction of the
    /// optional clauses, rounded as `Math.round(float)`.
    fn min_nr_should_match(min: f32, num_optional: usize) -> usize {
        if min >= 1.0 || min == 0.0 {
            return min as usize;
        }
        let v = min * num_optional as f32;
        // `Math.round(float)`: `floor(v + 0.5)`, as an `int`.
        (v + 0.5).floor().max(0.0) as usize
    }

    /// `rewrite(indexSearcher)`.
    ///
    /// # Errors
    /// A term dictionary that cannot be read.
    pub fn rewrite(&self, searcher: &IndexSearcher<'_, '_>) -> Result<Clause> {
        match self.terms.len() {
            0 => {
                return Ok(Clause::MatchNoDocs(
                    MatchNoDocsQuery::new().with_reason("CommonTermsQuery with no terms"),
                ))
            }
            1 => {
                let (f, t) = &self.terms[0];
                return Ok(Clause::Term(TermQuery::new(f.clone(), t.clone())));
            }
            _ => {}
        }
        // `collectTermStates`: each term's document frequency, summed over the
        // leaves that hold it; `None` where no leaf does.
        let mut doc_freqs: Vec<Option<i64>> = vec![None; self.terms.len()];
        for seg in searcher.segments() {
            for (i, (field, term)) in self.terms.iter().enumerate() {
                let Some(ft) = seg.fields.field(field) else {
                    continue;
                };
                if let Some(stats) = ft.try_seek_exact(term)? {
                    let df = doc_freqs[i].get_or_insert(0);
                    *df = df.saturating_add(i64::from(stats.doc_freq));
                }
            }
        }
        Ok(self.build_query(searcher.max_doc(), &doc_freqs))
    }

    /// `buildQuery(maxDoc, contextArray, queryTerms)`.
    fn build_query(&self, max_doc: i32, doc_freqs: &[Option<i64>]) -> Clause {
        let mut low = Vec::new();
        let mut high = Vec::new();
        // `(int) Math.ceil(maxTermFrequency * (float) maxDoc)`.
        let ceiling = i64::from(f64::from(self.max_term_frequency * max_doc as f32).ceil() as i32);
        for ((field, term), df) in self.terms.iter().zip(doc_freqs) {
            let q = Clause::Term(TermQuery::new(field.clone(), term.clone()));
            match df {
                None => low.push(q),
                Some(df) => {
                    let mtf = self.max_term_frequency;
                    if (mtf >= 1.0 && (*df as f32) > mtf) || *df > ceiling {
                        high.push(q);
                    } else {
                        low.push(q);
                    }
                }
            }
        }
        let low_occur = self.low_freq_occur;
        let mut high_occur = self.high_freq_occur;
        let mut low_msm = 0;
        let mut high_msm = 0;
        if low_occur == Occur::Should && !low.is_empty() {
            low_msm = Self::min_nr_should_match(self.low_freq_min_nr_should_match, low.len());
        }
        if high_occur == Occur::Should && !high.is_empty() {
            high_msm = Self::min_nr_should_match(self.high_freq_min_nr_should_match, high.len());
        }
        // Only frequent terms: they are what the document must have.
        if low.is_empty() && high_msm == 0 && high_occur != Occur::Must {
            high_occur = Occur::Must;
        }
        let group = |clauses: Vec<Clause>, occur: Occur, msm: usize| {
            let mut b = BooleanQuery::new();
            for c in clauses {
                match occur {
                    Occur::Must => b.must.push(c),
                    _ => b.should.push(c),
                }
            }
            b.minimum_should_match = msm;
            // `IndexSearcher.rewrite` runs to a fixpoint, so each group is
            // searched rewritten: a term named twice is one clause with the
            // boosts summed, which changes how many hits a conjunction
            // counts before it may stop counting.
            Clause::Boolean(Box::new(b)).rewrite()
        };
        let mut builder = BooleanQuery::new();
        if !low.is_empty() {
            builder.must.push(Clause::Boost(Box::new(BoostQuery::new(
                group(low, low_occur, low_msm),
                self.low_freq_boost,
            ))));
        }
        if !high.is_empty() {
            builder.should.push(Clause::Boost(Box::new(BoostQuery::new(
                group(high, high_occur, high_msm),
                self.high_freq_boost,
            ))));
        }
        Clause::Boolean(Box::new(builder))
    }
}

impl std::fmt::Display for CommonTermsQuery {
    /// `toString(field)` with no default field.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let need_parens = self.low_freq_min_nr_should_match > 0.0;
        if need_parens {
            f.write_str("(")?;
        }
        for (i, (field, term)) in self.terms.iter().enumerate() {
            write!(f, "{field}:{}", String::from_utf8_lossy(term))?;
            if i != self.terms.len() - 1 {
                f.write_str(", ")?;
            }
        }
        if need_parens {
            f.write_str(")")?;
        }
        if self.low_freq_min_nr_should_match > 0.0 || self.high_freq_min_nr_should_match > 0.0 {
            write!(
                f,
                "~({}{})",
                crate::explain::java_float(self.low_freq_min_nr_should_match),
                crate::explain::java_float(self.high_freq_min_nr_should_match)
            )?;
        }
        Ok(())
    }
}

impl From<CommonTermsQuery> for Clause {
    fn from(q: CommonTermsQuery) -> Self {
        Clause::Extended(Box::new(crate::extended_query::ExtendedQuery::CommonTerms(
            q,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_constructor_refuses_must_not_and_the_minimums_round_as_java() {
        assert!(CommonTermsQuery::new(Occur::MustNot, Occur::Should, 0.1).is_err());
        assert!(CommonTermsQuery::new(Occur::Should, Occur::MustNot, 0.1).is_err());
        assert_eq!(CommonTermsQuery::min_nr_should_match(0.0, 7), 0);
        assert_eq!(CommonTermsQuery::min_nr_should_match(2.0, 7), 2);
        assert_eq!(CommonTermsQuery::min_nr_should_match(2.7, 7), 2);
        assert_eq!(CommonTermsQuery::min_nr_should_match(0.5, 3), 2);
        assert_eq!(CommonTermsQuery::min_nr_should_match(0.5, 5), 3);
        assert_eq!(CommonTermsQuery::min_nr_should_match(0.25, 2), 1);
        let mut q = CommonTermsQuery::new(Occur::Should, Occur::Should, 0.1).unwrap();
        q.add("f", "a");
        q.add("f", "b");
        assert_eq!(q.to_string(), "f:a, f:b");
        q.low_freq_min_nr_should_match = 0.5;
        q.high_freq_min_nr_should_match = 2.0;
        assert_eq!(q.to_string(), "(f:a, f:b)~(0.52.0)");
    }

    /// The split: absent terms are rare, a count or a fraction of `maxDoc`
    /// decides the rest; only frequent terms are required.
    #[test]
    fn terms_split_by_document_frequency() {
        let mut q = CommonTermsQuery::new(Occur::Should, Occur::Should, 0.25).unwrap();
        for t in ["rare", "common", "absent"] {
            q.add("f", t);
        }
        let Clause::Boolean(b) = q.build_query(100, &[Some(5), Some(26), None]) else {
            panic!()
        };
        assert_eq!(b.must.len(), 1);
        assert_eq!(b.should.len(), 1);
        let mut only_high = CommonTermsQuery::new(Occur::Should, Occur::Should, 3.0).unwrap();
        only_high.add("f", "x");
        only_high.add("f", "y");
        let Clause::Boolean(b) = only_high.build_query(100, &[Some(4), Some(9)]) else {
            panic!()
        };
        assert!(b.must.is_empty());
        let Clause::Boost(inner) = &b.should[0] else {
            panic!()
        };
        let Clause::Boolean(high) = inner.inner.as_ref() else {
            panic!()
        };
        assert_eq!(high.must.len(), 2, "only frequent terms become required");
    }
}
