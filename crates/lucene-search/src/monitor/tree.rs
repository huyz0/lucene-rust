//! `org.apache.lucene.monitor.QueryTree` and `TermWeightor`: a query's
//! term tree, the input the presearcher indexes.
//!
//! Java's tree is an abstract class with anonymous subclasses for a term,
//! the any-token marker, and conjunction and disjunction nodes that sort
//! their children by weight. Here it is one enum with the same four
//! shapes; every method follows the Java method of the same name.

use std::fmt::Write as _;
use std::sync::Arc;

use crate::function::java_double;

/// One term of a query, as the presearcher keys it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Term {
    pub field: String,
    pub bytes: Vec<u8>,
}

/// `TermWeightor`: how much a term helps narrow a query, `>= 0` and as
/// `ToDoubleFunction<Term>`. Shared, so a tree can keep the one it was built
/// with.
#[derive(Clone)]
pub struct TermWeightor(Arc<dyn Fn(&Term) -> f64 + Send + Sync>);

impl std::fmt::Debug for TermWeightor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TermWeightor")
    }
}

impl TermWeightor {
    /// `TermWeightor.applyAsDouble(term)`.
    pub fn apply(&self, term: &Term) -> f64 {
        (self.0)(term)
    }

    /// `TermWeightor.combine(weightors...)`: the product of each weight.
    pub fn combine(weightors: Vec<TermWeightor>) -> TermWeightor {
        TermWeightor(Arc::new(move |t| {
            weightors.iter().fold(1.0, |r, w| r * w.apply(t))
        }))
    }

    /// `TermWeightor.fieldWeightor(weight, fields)`.
    pub fn field(weight: f64, fields: &[&str]) -> TermWeightor {
        let fields: Vec<String> = fields.iter().map(|f| (*f).to_string()).collect();
        TermWeightor(Arc::new(move |t| {
            if fields.contains(&t.field) {
                weight
            } else {
                1.0
            }
        }))
    }

    /// `TermWeightor.termWeightor(weight, terms)`.
    pub fn bytes(weight: f64, terms: &[&[u8]]) -> TermWeightor {
        let terms: Vec<Vec<u8>> = terms.iter().map(|t| t.to_vec()).collect();
        TermWeightor(Arc::new(move |t| {
            if terms.contains(&t.bytes) {
                weight
            } else {
                1.0
            }
        }))
    }

    /// `TermWeightor.termFreqWeightor(frequencies, n, k)`: `n / frequency + k`
    /// for a term the map counts, `1` otherwise. `text` is the term's UTF-8
    /// text, as `Term.text()`.
    pub fn term_freq(frequencies: Vec<(String, i32)>, n: f64, k: f64) -> TermWeightor {
        TermWeightor(Arc::new(move |t| {
            let text = String::from_utf8_lossy(&t.bytes);
            match frequencies.iter().find(|(s, _)| *s == text) {
                Some((_, count)) => (n / f64::from(*count)) + k,
                None => 1.0,
            }
        }))
    }

    /// `TermWeightor.lengthWeightor(a, k)`: four minus a length norm that
    /// falls with the term's length, the last entry for a term of 32 bytes or
    /// more. Each norm is `(float) (a * exp(-k * i))`, as Java casts it.
    pub fn length(a: f64, k: f64) -> TermWeightor {
        let norms: Vec<f64> = (0..32)
            .map(|i| f64::from((a * (-k * f64::from(i)).exp()) as f32))
            .collect();
        TermWeightor(Arc::new(move |t| {
            let len = t.bytes.len();
            if len >= 32 {
                4.0 - norms[31]
            } else {
                4.0 - norms[len]
            }
        }))
    }
}

impl Default for TermWeightor {
    /// `TermWeightor.DEFAULT`: `lengthWeightor(3, 0.3f)`.
    fn default() -> Self {
        TermWeightor::length(3.0, f64::from(0.3f32))
    }
}

/// `QueryTree`: a query's terms, weighted, as conjunctions and disjunctions.
#[derive(Debug, Clone)]
pub enum QueryTree {
    /// `QueryTree.term(field, term, weight)`.
    Term { term: Term, weight: f64 },
    /// `QueryTree.anyTerm(reason)`: a query no term can narrow; indexed under
    /// the any-token marker, weight `0`.
    AnyTerm { reason: String },
    /// A node a [`crate::monitor::handlers::CustomQueryHandler`] builds for a
    /// query no term names: the anonymous `QueryTree` of
    /// `RegexpQueryHandler`. `terms` are what it indexes, `description` its
    /// `toString` (without indentation).
    Custom {
        weight: f64,
        terms: Vec<Term>,
        description: String,
    },
    /// `ConjunctionQueryTree`: children sorted by descending weight.
    Conjunction(Vec<QueryTree>),
    /// `DisjunctionQueryTree`: children sorted by ascending weight.
    Disjunction(Vec<QueryTree>),
}

/// The field a query that no term can narrow is indexed under.
pub const ANYTOKEN_FIELD: &str = "__anytokenfield";
/// The term that marker holds.
pub const ANYTOKEN: &[u8] = b"__ANYTOKEN__";

/// A child's weight (`0` when it has none) as an integer ordered as
/// `f64::total_cmp` orders the weight -- the key `advance_phase` sorts by, so
/// each child's weight is computed once per sort, not once per comparison.
fn total_key(tree: &QueryTree) -> i64 {
    let bits = tree.weight().unwrap_or(0.0).to_bits() as i64;
    // `total_cmp`'s own transform: flip the magnitude bits of a negative
    bits ^ (((bits >> 63) as u64) >> 1) as i64
}

impl QueryTree {
    /// `QueryTree.term(term, weightor)`.
    pub fn term(term: Term, weightor: &TermWeightor) -> QueryTree {
        let weight = weightor.apply(&term);
        QueryTree::Term { term, weight }
    }

    /// `QueryTree.term(field, term, weight)`.
    pub fn term_weighted(field: &str, bytes: &[u8], weight: f64) -> QueryTree {
        QueryTree::Term {
            term: Term {
                field: field.to_string(),
                bytes: bytes.to_vec(),
            },
            weight,
        }
    }

    /// `QueryTree.anyTerm(reason)`.
    pub fn any_term(reason: impl Into<String>) -> QueryTree {
        QueryTree::AnyTerm {
            reason: reason.into(),
        }
    }

    /// `weight()`: a term's weight, a node's first child's (the heaviest of a
    /// conjunction, the lightest of a disjunction). Java throws for a term
    /// weight of `0` or less, and for a term, so that is checked here too.
    pub fn weight(&self) -> Result<f64, String> {
        match self {
            QueryTree::Term { weight, .. } => {
                if *weight <= 0.0 {
                    Err("Term weights must be greater than 0".to_string())
                } else {
                    Ok(*weight)
                }
            }
            QueryTree::AnyTerm { .. } => Ok(0.0),
            QueryTree::Custom { weight, .. } => Ok(*weight),
            QueryTree::Conjunction(children) | QueryTree::Disjunction(children) => {
                children[0].weight()
            }
        }
    }

    /// `collectTerms(termCollector)`: every term the tree would index, the
    /// first child's only for a conjunction, every child's for a disjunction.
    pub fn collect_terms(&self, out: &mut Vec<Term>) {
        match self {
            QueryTree::Term { term, .. } => out.push(term.clone()),
            QueryTree::Custom { terms, .. } => out.extend(terms.iter().cloned()),
            QueryTree::AnyTerm { .. } => out.push(Term {
                field: ANYTOKEN_FIELD.to_string(),
                bytes: ANYTOKEN.to_vec(),
            }),
            QueryTree::Conjunction(children) => children[0].collect_terms(out),
            QueryTree::Disjunction(children) => {
                for c in children {
                    c.collect_terms(out);
                }
            }
        }
    }

    /// `advancePhase(minWeight)`: moves to the next-most-weighted path below
    /// this node, if one is above `min_weight`. A conjunction advances its
    /// heaviest child, or drops it when the next child clears `min_weight`; a
    /// disjunction advances every child. `false` when nothing changed.
    pub fn advance_phase(&mut self, min_weight: f64) -> bool {
        match self {
            QueryTree::Term { .. } | QueryTree::AnyTerm { .. } | QueryTree::Custom { .. } => false,
            QueryTree::Conjunction(children) => {
                if children[0].advance_phase(min_weight) {
                    // heaviest first; each child's weight walked once
                    children.sort_by_cached_key(|c| std::cmp::Reverse(total_key(c)));
                    return true;
                }
                if children.len() == 1 {
                    return false;
                }
                if children[1].weight().unwrap_or(0.0) <= min_weight {
                    return false;
                }
                children.remove(0);
                true
            }
            QueryTree::Disjunction(children) => {
                let mut changed = false;
                for c in children.iter_mut() {
                    changed |= c.advance_phase(min_weight);
                }
                if changed {
                    children.sort_by_cached_key(total_key);
                }
                changed
            }
        }
    }

    /// `toString(depth)`: Java's indented layout, two spaces a level. A
    /// term weighted `0` or less is an error, as `weight()` throws for it.
    pub fn to_string_at(&self, depth: usize) -> Result<String, String> {
        let pad = " ".repeat(depth);
        Ok(match self {
            QueryTree::Term { term, weight } => {
                self.weight()?;
                format!(
                    "{pad}{}:{}^{}",
                    term.field,
                    String::from_utf8_lossy(&term.bytes),
                    java_double(*weight)
                )
            }
            QueryTree::AnyTerm { reason } => format!("{pad}ANY[{reason}]"),
            QueryTree::Custom { description, .. } => format!("{pad}{description}"),
            QueryTree::Conjunction(children) | QueryTree::Disjunction(children) => {
                let kind = if matches!(self, QueryTree::Conjunction(_)) {
                    "Conjunction"
                } else {
                    "Disjunction"
                };
                let mut sb = format!(
                    "{pad}{kind}[{}]^{}
",
                    children.len(),
                    java_double(self.weight()?)
                );
                for child in children {
                    let _ = writeln!(sb, "{}", child.to_string_at(depth + 2)?);
                }
                sb
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sort_key_orders_weights_as_total_cmp_does() {
        let custom = |weight: f64| QueryTree::Custom {
            weight,
            terms: Vec::new(),
            description: String::new(),
        };
        let ws = [
            3.5,
            -0.0,
            0.0,
            1e-300,
            -2.0,
            f64::INFINITY,
            f64::NAN,
            7.25,
            -f64::NAN,
        ];
        for a in ws {
            for b in ws {
                assert_eq!(
                    total_key(&custom(a)).cmp(&total_key(&custom(b))),
                    a.total_cmp(&b),
                    "{a} {b}"
                );
            }
        }
        // a node with no weight (an error) sorts as 0
        let bad = QueryTree::Term {
            term: Term {
                field: "f".into(),
                bytes: b"t".to_vec(),
            },
            weight: 0.0,
        };
        assert!(bad.weight().is_err());
        assert_eq!(total_key(&bad), total_key(&custom(0.0)));
    }
}
