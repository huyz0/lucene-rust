//! `org.apache.lucene.spatial.query`: `SpatialOperation`, `SpatialArgs`,
//! `SpatialArgsParser` and `UnsupportedSpatialOperation`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::spatial4j::shape::{Shape, SpatialRelation};
use crate::spatial4j::{Error, Result, SpatialContext};

/// `SpatialOperation`: the predicates a strategy may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialOperation {
    /// `BBoxIntersects`: the indexed shape's bounding box intersects.
    BBoxIntersects,
    /// `BBoxWithin` (alias `BBoxCoveredBy`).
    BBoxWithin,
    /// `Contains` (alias `Covers`).
    Contains,
    /// `Intersects`.
    Intersects,
    /// `Equals` (alias `IsEqualTo`).
    IsEqualTo,
    /// `Disjoint` (alias `IsDisjointTo`).
    IsDisjointTo,
    /// `Within` (aliases `IsWithin`, `CoveredBy`).
    IsWithin,
    /// `Overlaps`: intersects, but neither contains nor is within.
    Overlaps,
}

impl SpatialOperation {
    /// `values()`, in Java's registration order.
    pub const VALUES: [SpatialOperation; 8] = [
        SpatialOperation::BBoxIntersects,
        SpatialOperation::BBoxWithin,
        SpatialOperation::Contains,
        SpatialOperation::Intersects,
        SpatialOperation::IsEqualTo,
        SpatialOperation::IsDisjointTo,
        SpatialOperation::IsWithin,
        SpatialOperation::Overlaps,
    ];

    /// `getName()`.
    pub fn name(self) -> &'static str {
        match self {
            SpatialOperation::BBoxIntersects => "BBoxIntersects",
            SpatialOperation::BBoxWithin => "BBoxWithin",
            SpatialOperation::Contains => "Contains",
            SpatialOperation::Intersects => "Intersects",
            SpatialOperation::IsEqualTo => "Equals",
            SpatialOperation::IsDisjointTo => "Disjoint",
            SpatialOperation::IsWithin => "Within",
            SpatialOperation::Overlaps => "Overlaps",
        }
    }

    /// The names an operation is registered under (its name first).
    fn names(self) -> &'static [&'static str] {
        match self {
            SpatialOperation::BBoxIntersects => &["BBoxIntersects"],
            SpatialOperation::BBoxWithin => &["BBoxWithin", "BBoxCoveredBy"],
            SpatialOperation::Contains => &["Contains", "Covers"],
            SpatialOperation::Intersects => &["Intersects"],
            SpatialOperation::IsEqualTo => &["Equals", "IsEqualTo"],
            SpatialOperation::IsDisjointTo => &["Disjoint", "IsDisjointTo"],
            SpatialOperation::IsWithin => &["Within", "IsWithin", "CoveredBy"],
            SpatialOperation::Overlaps => &["Overlaps"],
        }
    }

    /// `SpatialOperation.get(v)`: by name or alias, exactly or in upper
    /// case.
    pub fn get(v: &str) -> Result<SpatialOperation> {
        let upper = v.to_uppercase();
        for key in [v, upper.as_str()] {
            for op in Self::VALUES {
                if op
                    .names()
                    .iter()
                    .any(|n| *n == key || n.to_uppercase() == key)
                {
                    return Ok(op);
                }
            }
        }
        Err(Error::IllegalArgument(format!("Unknown Operation: {v}")))
    }

    /// `evaluate(indexedShape, queryShape)`.
    pub fn evaluate(self, indexed: &dyn Shape, query: &dyn Shape) -> Result<bool> {
        Ok(match self {
            SpatialOperation::BBoxIntersects => indexed.bounding_box()?.relate(query)?.intersects(),
            SpatialOperation::BBoxWithin => {
                let bbox = indexed.bounding_box()?;
                bbox.relate(query)? == SpatialRelation::Within || bbox.equals(query)
            }
            SpatialOperation::Contains => {
                indexed.relate(query)? == SpatialRelation::Contains || indexed.equals(query)
            }
            SpatialOperation::Intersects => indexed.relate(query)?.intersects(),
            SpatialOperation::IsEqualTo => indexed.equals(query),
            SpatialOperation::IsDisjointTo => !indexed.relate(query)?.intersects(),
            SpatialOperation::IsWithin => {
                indexed.relate(query)? == SpatialRelation::Within || indexed.equals(query)
            }
            SpatialOperation::Overlaps => indexed.relate(query)? == SpatialRelation::Intersects,
        })
    }

    /// `new UnsupportedSpatialOperation(op)`: an
    /// `UnsupportedOperationException` with the operation's name.
    pub fn unsupported(self) -> Error {
        Error::UnsupportedSpatialOperation(self.name().to_string())
    }
}

impl fmt::Display for SpatialOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// `SpatialArgs`: an operation, a shape and the acceptable error.
#[derive(Debug, Clone)]
pub struct SpatialArgs {
    pub operation: SpatialOperation,
    pub shape: Arc<dyn Shape>,
    dist_err_pct: Option<f64>,
    dist_err: Option<f64>,
}

impl SpatialArgs {
    /// `SpatialArgs.DEFAULT_DISTERRPCT`.
    pub const DEFAULT_DISTERRPCT: f64 = 0.025;

    /// `new SpatialArgs(operation, shape)`.
    pub fn new(operation: SpatialOperation, shape: Arc<dyn Shape>) -> Self {
        SpatialArgs {
            operation,
            shape,
            dist_err_pct: None,
            dist_err: None,
        }
    }

    /// `calcDistanceFromErrPct(shape, distErrPct, ctx)`: `distErrPct` times
    /// the distance from the bounding box's center to its nearer corner
    /// (top for a center north of the equator); 0 for a point.
    pub fn calc_distance_from_err_pct(
        shape: &dyn Shape,
        dist_err_pct: f64,
        ctx: &Arc<SpatialContext>,
    ) -> Result<f64> {
        // Not `!(0.0..=0.5).contains(..)`: Java lets NaN through.
        #[allow(clippy::manual_range_contains)]
        if dist_err_pct < 0.0 || dist_err_pct > 0.5 {
            return Err(Error::IllegalArgument(format!(
                "distErrPct {} must be between [0 to 0.5]",
                crate::spatial4j::dstr(dist_err_pct)
            )));
        }
        if dist_err_pct == 0.0 || shape.as_point().is_some() {
            return Ok(0.0);
        }
        let bbox = shape.bounding_box()?;
        let ctr = bbox.center()?;
        let y = if ctr.y() >= 0.0 {
            bbox.max_y()
        } else {
            bbox.min_y()
        };
        let diagonal_dist = ctx.dist_calc().distance_xy(&*ctr, bbox.max_x(), y)?;
        Ok(diagonal_dist * dist_err_pct)
    }

    /// `resolveDistErr(ctx, defaultDistErrPct)`.
    pub fn resolve_dist_err(
        &self,
        ctx: &Arc<SpatialContext>,
        default_dist_err_pct: f64,
    ) -> Result<f64> {
        if let Some(d) = self.dist_err {
            return Ok(d);
        }
        let pct = self.dist_err_pct.unwrap_or(default_dist_err_pct);
        Self::calc_distance_from_err_pct(&*self.shape, pct, ctx)
    }

    /// `validate()`.
    pub fn validate(&self) -> Result<()> {
        if self.dist_err.is_some() && self.dist_err_pct.is_some() {
            return Err(Error::IllegalArgument(
                "Only distErr or distErrPct can be specified.".into(),
            ));
        }
        Ok(())
    }

    /// `getDistErrPct()`.
    pub fn dist_err_pct(&self) -> Option<f64> {
        self.dist_err_pct
    }

    /// `setDistErrPct(distErrPct)`: a `null` is ignored.
    pub fn set_dist_err_pct(&mut self, dist_err_pct: Option<f64>) {
        if dist_err_pct.is_some() {
            self.dist_err_pct = dist_err_pct;
        }
    }

    /// `getDistErr()`.
    pub fn dist_err(&self) -> Option<f64> {
        self.dist_err
    }

    /// `setDistErr(distErr)`.
    pub fn set_dist_err(&mut self, dist_err: Option<f64>) {
        self.dist_err = dist_err;
    }
}

impl fmt::Display for SpatialArgs {
    /// `SpatialArgsParser.writeSpatialArgs(args)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}", self.operation.name(), self.shape)?;
        if let Some(p) = self.dist_err_pct {
            write!(
                f,
                " distErrPct={}%",
                crate::spatial4j::java_format_fixed(p * 100.0, 2)
            )?;
        }
        if let Some(d) = self.dist_err {
            write!(f, " distErr={}", crate::spatial4j::dstr(d))?;
        }
        f.write_str(")")
    }
}

/// `String.trim()`: strips characters at or below the space.
fn java_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c <= ' ')
}

/// `SpatialArgsParser`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpatialArgsParser;

impl SpatialArgsParser {
    /// `DIST_ERR_PCT`.
    pub const DIST_ERR_PCT: &'static str = "distErrPct";
    /// `DIST_ERR`.
    pub const DIST_ERR: &'static str = "distErr";

    /// `parse(v, ctx)`: e.g. `Intersects(ENVELOPE(-10,-8,22,20))
    /// distErrPct=0.025`; the shape is WKT.
    pub fn parse(&self, v: &str, ctx: &Arc<SpatialContext>) -> Result<SpatialArgs> {
        let chars: Vec<char> = v.chars().collect();
        let idx = chars.iter().position(|&c| c == '(');
        let edx = chars.iter().rposition(|&c| c == ')');
        let (idx, edx) = match (idx, edx) {
            (Some(i), Some(e)) if i <= e => (i, e),
            _ => {
                return Err(Error::Parse {
                    message: format!("missing parens: {v}"),
                    offset: -1,
                })
            }
        };
        let head: String = chars[..idx].iter().collect();
        let op = SpatialOperation::get(java_trim(&head))?;
        let body: String = chars[idx + 1..edx].iter().collect();
        let body = java_trim(&body);
        if body.is_empty() {
            return Err(Error::Parse {
                message: format!("missing body : {v}"),
                offset: idx as i32 + 1,
            });
        }
        let shape = ctx.wkt_reader().parse(body)?;
        let mut args = SpatialArgs::new(op, shape);
        if chars.len() > edx + 1 {
            let rest: String = chars[edx + 1..].iter().collect();
            let rest = java_trim(&rest);
            if !rest.is_empty() {
                let mut aa = Self::parse_map(rest);
                args.set_dist_err_pct(Self::read_double(aa.remove(Self::DIST_ERR_PCT))?);
                args.set_dist_err(Self::read_double(aa.remove(Self::DIST_ERR))?);
                if !aa.is_empty() {
                    let entries: Vec<String> = aa.iter().map(|(k, v)| format!("{k}={v}")).collect();
                    return Err(Error::IllegalArgument(format!(
                        "unused parameters: {{{}}}",
                        entries.join(", ")
                    )));
                }
            }
        }
        args.validate()?;
        Ok(args)
    }

    /// `readDouble(v)`: `Double.valueOf`.
    fn read_double(v: Option<String>) -> Result<Option<f64>> {
        match v {
            None => Ok(None),
            Some(s) => crate::spatial4j::wkt::java_parse_double(java_trim(&s))
                .map(Some)
                .map_err(|m| {
                    Error::NumberFormat(
                        m.trim_start_matches("java.lang.NumberFormatException: ")
                            .to_string(),
                    )
                }),
        }
    }

    /// `parseMap(body)`: whitespace-separated `name=value` (or `name`,
    /// short for `name=name`). Java keeps them in a `HashMap`; a leftover
    /// list is printed in name order here.
    pub fn parse_map(body: &str) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        for a in body.split([' ', '\n', '\t']).filter(|t| !t.is_empty()) {
            match a.find('=') {
                Some(idx) if idx > 0 => {
                    map.insert(a[..idx].to_string(), a[idx + 1..].to_string());
                }
                _ => {
                    map.insert(a.to_string(), a.to_string());
                }
            }
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_names_and_aliases() {
        for op in SpatialOperation::VALUES {
            for n in op.names() {
                assert_eq!(SpatialOperation::get(n).unwrap(), op);
                assert_eq!(SpatialOperation::get(&n.to_uppercase()).unwrap(), op);
            }
            assert_eq!(op.to_string(), op.name());
            assert_eq!(op.unsupported().to_string(), op.name());
        }
        assert_eq!(
            SpatialOperation::get("nope").unwrap_err().to_string(),
            "Unknown Operation: nope"
        );
        assert_eq!(
            SpatialOperation::get("covers").unwrap(),
            SpatialOperation::Contains
        );
    }

    #[test]
    fn args_accessors() {
        let ctx = SpatialContext::geo_context();
        let p = ctx.point_xy(1.0, 2.0).unwrap();
        let mut a = SpatialArgs::new(SpatialOperation::Intersects, p);
        a.set_dist_err_pct(None);
        assert_eq!(a.dist_err_pct(), None);
        a.set_dist_err(Some(1.5));
        assert_eq!(a.dist_err(), Some(1.5));
        assert_eq!(a.resolve_dist_err(&ctx, 0.1).unwrap(), 1.5);
        assert_eq!(a.to_string(), "Intersects(Pt(x=1.0,y=2.0) distErr=1.5)");
        a.set_dist_err_pct(Some(0.25));
        assert!(a.validate().is_err());
        assert!(SpatialArgs::calc_distance_from_err_pct(&*a.shape, 0.6, &ctx).is_err());
        let m = SpatialArgsParser::parse_map("a=b c\td=");
        assert_eq!(m.get("c").map(String::as_str), Some("c"));
        assert_eq!(m.get("d").map(String::as_str), Some(""));
    }
}
