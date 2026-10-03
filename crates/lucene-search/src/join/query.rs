//! The block-join queries as plain descriptions of the Java queries, carried
//! by the scorer tree's [`Clause::Extended`]; how each runs per segment is
//! `exec::join`.

use std::fmt;
use std::sync::Arc;

use super::BitSetProducer;
use crate::extended_query::ExtendedQuery;
use crate::query::Clause;
use crate::{Error, Result};

/// `ScoreMode` (`org.apache.lucene.search.join`): how the scores of a
/// parent's matching children make the parent's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreMode {
    /// Every parent scores `0`; children are not scored.
    None,
    /// The mean of the children's scores (summed in `double`).
    Avg,
    /// The highest child score.
    Max,
    /// The sum of the children's scores (in `double`).
    Total,
    /// The lowest child score.
    Min,
}

impl fmt::Display for ScoreMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `Enum.toString()`, as `Explanation`'s description prints it.
        f.write_str(match self {
            ScoreMode::None => "None",
            ScoreMode::Avg => "Avg",
            ScoreMode::Max => "Max",
            ScoreMode::Total => "Total",
            ScoreMode::Min => "Min",
        })
    }
}

/// A producer compared by its identity (`BitSetProducer.equals`), so the
/// queries holding one are `Query.equals` to each other.
fn same(a: &Arc<dyn BitSetProducer>, b: &Arc<dyn BitSetProducer>) -> bool {
    Arc::ptr_eq(a, b) || a.key() == b.key()
}

/// `ToParentBlockJoinQuery`: the parents of the children `child` matches,
/// each scored from its matching children by `score_mode`. `child` must
/// never match a parent (Lucene throws `IllegalStateException` when it finds
/// one; so does this, as [`Error::IllegalState`]).
#[derive(Clone)]
pub struct ToParentBlockJoinQuery {
    pub child: Box<Clause>,
    pub parents: Arc<dyn BitSetProducer>,
    pub score_mode: ScoreMode,
}

impl ToParentBlockJoinQuery {
    /// `new ToParentBlockJoinQuery(childQuery, parentsFilter, scoreMode)`.
    pub fn new(
        child: impl Into<Clause>,
        parents: Arc<dyn BitSetProducer>,
        score_mode: ScoreMode,
    ) -> Self {
        Self {
            child: Box::new(child.into()),
            parents,
            score_mode,
        }
    }
}

impl fmt::Debug for ToParentBlockJoinQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ToParentBlockJoinQuery({:?}, {}, {})",
            self.child,
            self.parents.key(),
            self.score_mode
        )
    }
}

impl PartialEq for ToParentBlockJoinQuery {
    fn eq(&self, o: &Self) -> bool {
        self.child == o.child && same(&self.parents, &o.parents) && self.score_mode == o.score_mode
    }
}

/// `ToChildBlockJoinQuery`: the children of the parents `parent` matches,
/// each scored its parent's score. `parent` must match parents only
/// (Lucene's `IllegalStateException`, here [`Error::IllegalState`]).
#[derive(Clone)]
pub struct ToChildBlockJoinQuery {
    pub parent: Box<Clause>,
    pub parents: Arc<dyn BitSetProducer>,
}

impl ToChildBlockJoinQuery {
    /// `new ToChildBlockJoinQuery(parentQuery, parentsFilter)`.
    pub fn new(parent: impl Into<Clause>, parents: Arc<dyn BitSetProducer>) -> Self {
        Self {
            parent: Box::new(parent.into()),
            parents,
        }
    }
}

impl fmt::Debug for ToChildBlockJoinQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ToChildBlockJoinQuery({:?}, {})",
            self.parent,
            self.parents.key()
        )
    }
}

impl PartialEq for ToChildBlockJoinQuery {
    fn eq(&self, o: &Self) -> bool {
        self.parent == o.parent && same(&self.parents, &o.parents)
    }
}

/// `ParentChildrenBlockJoinQuery`: the children of one parent document
/// (`parent_doc`, a reader-wide doc id) that `child` matches, scored as
/// `child` scores them -- what an inner-hits fetch runs per hit.
#[derive(Clone)]
pub struct ParentChildrenBlockJoinQuery {
    pub parents: Arc<dyn BitSetProducer>,
    pub child: Box<Clause>,
    pub parent_doc: i32,
}

impl ParentChildrenBlockJoinQuery {
    /// `new ParentChildrenBlockJoinQuery(parentFilter, childQuery, parentDocId)`.
    pub fn new(
        parents: Arc<dyn BitSetProducer>,
        child: impl Into<Clause>,
        parent_doc: i32,
    ) -> Self {
        Self {
            parents,
            child: Box::new(child.into()),
            parent_doc,
        }
    }
}

impl fmt::Debug for ParentChildrenBlockJoinQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParentChildrenBlockJoinQuery({}, {:?}, {})",
            self.parents.key(),
            self.child,
            self.parent_doc
        )
    }
}

impl PartialEq for ParentChildrenBlockJoinQuery {
    fn eq(&self, o: &Self) -> bool {
        same(&self.parents, &o.parents) && self.child == o.child && self.parent_doc == o.parent_doc
    }
}

/// `ParentsChildrenBlockJoinQuery`'s `scoreCombiner`: how a hit's parent and
/// child scores combine (`BinaryOperator<Float>`, `Float::sum` by default).
#[derive(Clone)]
pub enum ScoreCombiner {
    /// `Float::sum`.
    Sum,
    /// Any other operator, identified by `name` (`Query.equals` compares the
    /// operator object; two custom combiners are equal when their names are).
    Custom {
        name: String,
        combine: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>,
    },
}

impl ScoreCombiner {
    pub(crate) fn apply(&self, parent: f32, child: f32) -> f32 {
        match self {
            ScoreCombiner::Sum => parent + child,
            ScoreCombiner::Custom { combine, .. } => combine(parent, child),
        }
    }

    fn name(&self) -> &str {
        match self {
            ScoreCombiner::Sum => "sum",
            ScoreCombiner::Custom { name, .. } => name,
        }
    }
}

/// `ParentsChildrenBlockJoinQuery.DEFAULT_CHILD_LIMIT_PER_PARENT`.
pub const DEFAULT_CHILD_LIMIT_PER_PARENT: i32 = i32::MAX;

/// `ParentsChildrenBlockJoinQuery`: the children `child` matches whose
/// parent `parent` matches, at most `child_limit_per_parent` per parent, each
/// scored `combiner(parentScore, childScore)` (`1.0` when scores are not
/// needed).
///
/// Lucene's `rewrite` rebuilds the query with the default combiner whenever
/// a sub-query rewrites (it drops a custom `scoreCombiner`); this port keeps
/// the combiner. Its weight also refuses to be asked for the same segment
/// twice (intra-segment concurrency); this one has no such state.
#[derive(Clone)]
pub struct ParentsChildrenBlockJoinQuery {
    pub parents: Arc<dyn BitSetProducer>,
    pub parent: Box<Clause>,
    pub child: Box<Clause>,
    pub child_limit_per_parent: i32,
    pub combiner: ScoreCombiner,
}

impl ParentsChildrenBlockJoinQuery {
    /// `new ParentsChildrenBlockJoinQuery(parentFilter, parentQuery,
    /// childQuery, childLimitPerParent)`, with `Float::sum`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a limit below 1, as Java's constructor.
    pub fn new(
        parents: Arc<dyn BitSetProducer>,
        parent: impl Into<Clause>,
        child: impl Into<Clause>,
        child_limit_per_parent: i32,
    ) -> Result<Self> {
        if child_limit_per_parent <= 0 {
            return Err(Error::IllegalArgument(format!(
                "childLimitPerParent must be > 0, got {child_limit_per_parent}"
            )));
        }
        Ok(Self {
            parents,
            parent: Box::new(parent.into()),
            child: Box::new(child.into()),
            child_limit_per_parent,
            combiner: ScoreCombiner::Sum,
        })
    }

    /// The same query combining scores with `combiner`.
    pub fn with_combiner(mut self, combiner: ScoreCombiner) -> Self {
        self.combiner = combiner;
        self
    }
}

impl fmt::Debug for ParentsChildrenBlockJoinQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParentsChildrenBlockJoinQuery({}, {:?}, {:?}, {}, {})",
            self.parents.key(),
            self.parent,
            self.child,
            self.child_limit_per_parent,
            self.combiner.name()
        )
    }
}

impl PartialEq for ParentsChildrenBlockJoinQuery {
    fn eq(&self, o: &Self) -> bool {
        same(&self.parents, &o.parents)
            && self.parent == o.parent
            && self.child == o.child
            && self.child_limit_per_parent == o.child_limit_per_parent
            && self.combiner.name() == o.combiner.name()
    }
}

macro_rules! into_clause {
    ($($ty:ident => $variant:ident),* $(,)?) => {
        $(
            impl From<$ty> for Clause {
                fn from(q: $ty) -> Self {
                    Clause::Extended(Box::new(ExtendedQuery::$variant(q)))
                }
            }
        )*
    };
}

into_clause! {
    ToParentBlockJoinQuery => ToParentBlockJoin,
    ToChildBlockJoinQuery => ToChildBlockJoin,
    ParentChildrenBlockJoinQuery => ParentChildrenBlockJoin,
    ParentsChildrenBlockJoinQuery => ParentsChildrenBlockJoin,
}
