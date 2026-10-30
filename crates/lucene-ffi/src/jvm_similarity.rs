//! The searcher's per-field similarities in front of a query blob: OpenSearch's
//! `SimilarityService.PerFieldSimilarity` (a `PerFieldSimilarityWrapper`)
//! unwrapped, for every field the query scores, to the Lucene similarity its
//! mapping names -- `BM25` with its parameters, `boolean`, `DFR`, `IB`,
//! `DFI`, `LMDirichlet`, `LMJelinekMercer`. A field the table leaves out
//! scores with the default `BM25Similarity`.
//!
//! ```text
//! QUERY_SIMILARITY (4), count: i32, then per field:
//!   field: bytes, kind: u8, then by kind:
//!     0 BM25             k1: f32, b: f32, discount_overlaps: u8
//!     1 boolean
//!     2 DFR              basic_model: u8 (G, IF, In, Ine), after_effect: u8 (B, L),
//!                        normalization, discount_overlaps: u8
//!     3 IB               distribution: u8 (LL, SPL), lambda: u8 (DF, TTF),
//!                        normalization, discount_overlaps: u8
//!     4 DFI              independence: u8 (standardized, saturated, chi-squared),
//!                        discount_overlaps: u8
//!     5 LMDirichlet      mu: f32, discount_overlaps: u8
//!     6 LMJelinekMercer  lambda: f32, discount_overlaps: u8
//!   normalization: u8 (0 none, 1 H1, 2 H2, 3 H3, 4 Z), then its f32 unless none
//! ```
//!
//! The LM similarities use Lucene's `DefaultCollectionModel`, the only one
//! OpenSearch builds.

use std::sync::Arc;

use lucene_search::similarities::{
    AfterEffect, BasicModel, Bm25Similarity, BooleanSimilarity, CollectionModel, DfiSimilarity,
    DfrSimilarity, Distribution, IbSimilarity, Independence, Lambda, LmDirichletSimilarity,
    LmJelinekMercerSimilarity, Normalization, PerFieldSimilarity, Similarity,
};

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader::Cursor;

/// Blob tag: a similarity table, then the rest of the request.
pub const QUERY_SIMILARITY: u8 = 4;

/// At most this many fields in a table: a query scores at most its node
/// limit's worth.
const MAX_FIELDS: usize = 1024;

const BM25: u8 = 0;
const BOOLEAN: u8 = 1;
const DFR: u8 = 2;
const IB: u8 = 3;
const DFI: u8 = 4;
const LM_DIRICHLET: u8 = 5;
const LM_JELINEK_MERCER: u8 = 6;

fn bad(message: impl Into<String>) -> FfiStatus {
    set_last_error(format!("similarity table: {}", message.into()));
    FfiStatus::InvalidArgument
}

/// Reads a table after its tag.
pub(crate) fn decode(c: &mut Cursor<'_>) -> Result<PerFieldSimilarity, FfiStatus> {
    let count = c.len()?;
    if count > MAX_FIELDS {
        return Err(bad(format!("{count} fields")));
    }
    let mut per_field = PerFieldSimilarity::new(Arc::new(Bm25Similarity::default()));
    for _ in 0..count {
        let field = std::str::from_utf8(c.bytes()?).map_err(|_| FfiStatus::InvalidUtf8)?;
        per_field = per_field.with_field(field, similarity(c)?);
    }
    Ok(per_field)
}

fn similarity(c: &mut Cursor<'_>) -> Result<Arc<dyn Similarity>, FfiStatus> {
    let kind = c.u8()?;
    Ok(match kind {
        BM25 => {
            let (k1, b) = (f32_of(c)?, f32_of(c)?);
            Arc::new(Bm25Similarity::new(k1, b, discount(c)?).map_err(bad)?)
        }
        BOOLEAN => Arc::new(BooleanSimilarity),
        DFR => {
            let basic = match c.u8()? {
                0 => BasicModel::G,
                1 => BasicModel::IF,
                2 => BasicModel::In,
                3 => BasicModel::Ine,
                m => return Err(bad(format!("basic model {m}"))),
            };
            let after = match c.u8()? {
                0 => AfterEffect::B,
                1 => AfterEffect::L,
                a => return Err(bad(format!("after effect {a}"))),
            };
            let normalization = normalization(c)?;
            Arc::new(
                DfrSimilarity::with_discount_overlaps(basic, after, normalization, discount(c)?)
                    .map_err(bad)?,
            )
        }
        IB => {
            let distribution = match c.u8()? {
                0 => Distribution::LL,
                1 => Distribution::SPL,
                d => return Err(bad(format!("distribution {d}"))),
            };
            let lambda = match c.u8()? {
                0 => Lambda::DF,
                1 => Lambda::TTF,
                l => return Err(bad(format!("lambda {l}"))),
            };
            let normalization = normalization(c)?;
            Arc::new(
                IbSimilarity::with_discount_overlaps(
                    distribution,
                    lambda,
                    normalization,
                    discount(c)?,
                )
                .map_err(bad)?,
            )
        }
        DFI => {
            let independence = match c.u8()? {
                0 => Independence::Standardized,
                1 => Independence::Saturated,
                2 => Independence::ChiSquared,
                i => return Err(bad(format!("independence {i}"))),
            };
            Arc::new(DfiSimilarity::with_discount_overlaps(
                independence,
                discount(c)?,
            ))
        }
        LM_DIRICHLET => {
            let mu = f32_of(c)?;
            Arc::new(
                LmDirichletSimilarity::new(CollectionModel::Default, discount(c)?, mu)
                    .map_err(bad)?,
            )
        }
        LM_JELINEK_MERCER => {
            let lambda = f32_of(c)?;
            Arc::new(
                LmJelinekMercerSimilarity::new(CollectionModel::Default, discount(c)?, lambda)
                    .map_err(bad)?,
            )
        }
        k => return Err(bad(format!("unknown similarity {k}"))),
    })
}

fn normalization(c: &mut Cursor<'_>) -> Result<Normalization, FfiStatus> {
    Ok(match c.u8()? {
        0 => Normalization::None,
        1 => Normalization::H1(f32_of(c)?),
        2 => Normalization::H2(f32_of(c)?),
        3 => Normalization::H3(f32_of(c)?),
        4 => Normalization::Z(f32_of(c)?),
        n => return Err(bad(format!("normalization {n}"))),
    })
}

fn f32_of(c: &mut Cursor<'_>) -> Result<f32, FfiStatus> {
    Ok(f32::from_bits(c.i32()? as u32))
}

fn discount(c: &mut Cursor<'_>) -> Result<bool, FfiStatus> {
    match c.u8()? {
        0 => Ok(false),
        1 => Ok(true),
        d => Err(bad(format!("discount_overlaps {d}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = (entries.len() as i32).to_le_bytes().to_vec();
        for (field, spec) in entries {
            out.extend_from_slice(&(field.len() as i32).to_le_bytes());
            out.extend_from_slice(field.as_bytes());
            out.extend_from_slice(spec);
        }
        out
    }

    fn f(v: f32) -> [u8; 4] {
        v.to_bits().to_le_bytes()
    }

    #[test]
    fn every_kind_decodes_to_its_similarity() {
        let bm25 = [&[BM25][..], &f(2.0), &f(0.5), &[0]].concat();
        let dfr = [&[DFR, 0, 1, 2][..], &f(3.0), &[1]].concat();
        let ib = [&[IB, 1, 1, 4][..], &f(0.3), &[1]].concat();
        let lmd = [&[LM_DIRICHLET][..], &f(1000.0), &[1]].concat();
        let lmj = [&[LM_JELINEK_MERCER][..], &f(0.3), &[1]].concat();
        let blob = table(&[
            ("a", &bm25),
            ("b", &[BOOLEAN]),
            ("c", &dfr),
            ("d", &ib),
            ("e", &[DFI, 2, 1]),
            ("f", &lmd),
            ("g", &lmj),
        ]);
        let sims = decode(&mut Cursor::new(&blob)).unwrap();
        let names: Vec<&str> = ["a", "b", "c", "d", "e", "f", "g", "other"]
            .iter()
            .map(|field| sims.get(field).simple_name())
            .collect();
        assert_eq!(
            names,
            [
                "BM25Similarity",
                "BooleanSimilarity",
                "DFRSimilarity",
                "IBSimilarity",
                "DFISimilarity",
                "LMDirichletSimilarity",
                "LMJelinekMercerSimilarity",
                "BM25Similarity",
            ]
        );
        use lucene_index::similarity::NormSimilarity as _;
        let a = sims.get("a").as_bm25("a").unwrap();
        assert_eq!((a.k1(), a.b(), a.discount_overlaps()), (2.0, 0.5, false));
        assert!(!sims.is_default_bm25());
    }

    #[test]
    fn malformed_tables_are_refused() {
        // An unknown kind, basic model, after effect, normalization,
        // distribution or independence measure.
        for spec in [
            &[9u8][..],
            &[DFR, 7, 0, 0, 1],
            &[DFR, 0, 5, 0, 1],
            &[DFR, 0, 0, 9, 1],
            &[IB, 2, 0, 0, 1],
            &[DFI, 3, 1],
        ] {
            assert!(
                decode(&mut Cursor::new(&table(&[("a", spec)]))).is_err(),
                "{spec:?}"
            );
        }
        // A negative mu, lambda out of range, discount flag not 0/1, a short table.
        let lmd = [&[LM_DIRICHLET][..], &f(-1.0), &[1]].concat();
        let lmj = [&[LM_JELINEK_MERCER][..], &f(1.5), &[1]].concat();
        let flag = [&[DFI, 0][..], &[2]].concat();
        for spec in [lmd, lmj, flag] {
            assert!(decode(&mut Cursor::new(&table(&[("a", &spec)]))).is_err());
        }
        assert!(decode(&mut Cursor::new(&(MAX_FIELDS as i32 + 1).to_le_bytes())).is_err());
        assert!(decode(&mut Cursor::new(&table(&[("a", &[BM25, 0])]))).is_err());
    }
}
