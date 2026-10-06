//! `QueryValueSource` and its `QueryDocValues`: a query's score where it
//! matches, a default elsewhere.

use crate::exec::{BoxScorer, LeafContext, Mode};
use crate::explain::describe_clause;
use crate::function::docvalues::{Float, FloatDocValues};
use crate::function::{
    java_float, out_of_order, BoxValues, FunctionContext, MutableValue, ObjectVal, ValueLeaf,
    ValueSource,
};
use crate::query::Clause;
use crate::Result;

/// `QueryValueSource`: the score of `q` (`ScoreMode.COMPLETE`, boost `1`)
/// for a document it matches, `defVal` for any other.
///
/// The query scores with the reader-wide statistics of the leaf it is read
/// on ([`ValueLeaf`]'s, which a search gathers for every query inside a
/// function query, as `createWeight`'s `searcher.createWeight` does).
#[derive(Debug, Clone)]
pub struct QueryValueSource {
    q: Clause,
    /// `searcher.rewrite(q)`.
    rewritten: Clause,
    def_val: f32,
}

impl QueryValueSource {
    pub fn new(q: impl Into<Clause>, def_val: f32) -> Self {
        let q = q.into();
        Self {
            rewritten: q.clone().rewrite(),
            q,
            def_val,
        }
    }

    /// `getQuery()`.
    pub fn query(&self) -> &Clause {
        &self.q
    }

    /// `getDefaultValue()`.
    pub fn default_value(&self) -> f32 {
        self.def_val
    }
}

impl ValueSource for QueryValueSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Float(QueryDocValues {
            ctx: leaf.ctx,
            q: self.q.clone(),
            rewritten: self.rewritten.clone(),
            def_val: self.def_val,
            scorer: None,
            built: false,
            this_doc_matches: None,
            last_doc_requested: -1,
        })))
    }
    fn description(&self) -> String {
        format!(
            "query({},def={})",
            describe_clause(&self.q),
            java_float(self.def_val)
        )
    }
    fn queries(&self) -> Vec<&Clause> {
        vec![&self.q]
    }
}

/// `QueryDocValues`: the query's scorer, built on the first document asked
/// for and advanced with the documents (its two-phase check run once per
/// document).
struct QueryDocValues<'a> {
    ctx: LeafContext<'a>,
    q: Clause,
    rewritten: Clause,
    def_val: f32,
    scorer: Option<BoxScorer<'a>>,
    built: bool,
    this_doc_matches: Option<bool>,
    last_doc_requested: i32,
}

impl FloatDocValues for QueryDocValues<'_> {
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        if self.exists(doc)? {
            if let Some(s) = &mut self.scorer {
                return s.score();
            }
        }
        Ok(self.def_val)
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        if doc < self.last_doc_requested {
            return Err(out_of_order(self.last_doc_requested, doc));
        }
        self.last_doc_requested = doc;
        if !self.built {
            self.scorer =
                crate::exec::build::build(&self.ctx, &self.rewritten, 1.0, Mode::Complete, false)?;
            self.built = true;
            self.this_doc_matches = None;
        }
        let Some(s) = &mut self.scorer else {
            return Ok(false);
        };
        if s.doc_id() < doc {
            s.advance(doc)?;
            self.this_doc_matches = None;
        }
        if s.doc_id() == doc {
            if self.this_doc_matches.is_none() {
                self.this_doc_matches = Some(!s.two_phase() || s.matches()?);
            }
            return Ok(self.this_doc_matches == Some(true));
        }
        Ok(false)
    }
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(ObjectVal::Float(self.float_val(doc)?))
    }
    /// `mval.value = scorer.score()` with `exists` true where the query
    /// matches, `defVal` with `exists` false elsewhere.
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        *out = if self.exists(doc)? {
            let value = match &mut self.scorer {
                Some(s) => s.score()?,
                None => self.def_val,
            };
            MutableValue::Float {
                value,
                exists: true,
            }
        } else {
            MutableValue::Float {
                value: self.def_val,
                exists: false,
            }
        };
        Ok(())
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        Ok(format!(
            "query({},def={})={}",
            describe_clause(&self.q),
            java_float(self.def_val),
            java_float(self.float_val(doc)?)
        ))
    }
}
