//! `org.apache.lucene.index.PointValues.Relation`: how a cell of a points
//! tree relates to a query shape.
//!
//! It lives here rather than beside the points reader because the geo
//! relations (`lucene_util::geo`, `Component2D.relate`) answer in it too, and
//! `lucene-util` is the lowest crate both reach; `lucene_codecs::points`
//! re-exports it under its old path.

/// Port of `org.apache.lucene.index.PointValues.Relation`. The declaration
/// order is Java's, so `as u8` is Java's `ordinal()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Relation {
    /// Every point in the cell matches -- no per-point check needed.
    CellInsideQuery,
    /// No point in the cell can match -- the whole subtree is skipped.
    CellOutsideQuery,
    /// The cell straddles the query boundary -- descend / check per point.
    CellCrossesQuery,
}

impl Relation {
    /// `Relation.values()[ordinal]`.
    pub fn from_ordinal(ordinal: u8) -> Option<Relation> {
        match ordinal {
            0 => Some(Relation::CellInsideQuery),
            1 => Some(Relation::CellOutsideQuery),
            2 => Some(Relation::CellCrossesQuery),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinals_are_javas() {
        for r in [
            Relation::CellInsideQuery,
            Relation::CellOutsideQuery,
            Relation::CellCrossesQuery,
        ] {
            assert_eq!(Relation::from_ordinal(r as u8), Some(r));
        }
        assert_eq!(Relation::CellCrossesQuery as u8, 2);
        assert_eq!(Relation::from_ordinal(3), None);
    }
}
