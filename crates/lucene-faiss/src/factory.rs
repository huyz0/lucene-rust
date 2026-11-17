//! Faiss index factory strings and index parameters, for the index types
//! this port builds and searches.
//!
//! Lucene's `FaissKnnVectorsFormat` takes a factory string (`description`,
//! default `IDMap,HNSW16`) and a parameter string (`indexParams`, default
//! `efConstruction=100`). Supported here:
//!
//! * `[IDMap,]Flat`
//! * `[IDMap,]HNSW<M>[,Flat]`
//! * `[IDMap,]IVF<nlist>,Flat`
//!
//! Any other string is a clear [`Error::Description`], never a silent
//! substitute: PQ, scalar quantization, `RFlat` refinement, PCA and the rest
//! of Faiss's factory language are not built.

use crate::error::{Error, Result};

/// The shape a factory string names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `Flat`: exhaustive search.
    Flat,
    /// `HNSW<M>[,Flat]`: a graph of `m` links per vertex (`2 * m` at level 0).
    Hnsw {
        /// Links per vertex.
        m: usize,
    },
    /// `IVF<nlist>,Flat`: `nlist` inverted lists over flat codes.
    Ivf {
        /// Number of lists.
        nlist: usize,
    },
}

/// A parsed factory string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Description {
    /// Whether the index is wrapped in `IndexIDMap` (`IDMap,` prefix).
    pub id_map: bool,
    /// The index named.
    pub kind: Kind,
}

const SUPPORTED: &str = "[IDMap,]Flat, [IDMap,]HNSW<M>[,Flat] or [IDMap,]IVF<nlist>,Flat";

fn refuse<T>(description: &str, reason: impl Into<String>) -> Result<T> {
    Err(Error::Description {
        description: description.to_string(),
        reason: reason.into(),
    })
}

fn number(description: &str, token: &str, prefix: &str) -> Result<usize> {
    match token[prefix.len()..].parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => refuse(
            description,
            format!("[{token}] needs a positive number after {prefix}"),
        ),
    }
}

/// Parses a factory string.
///
/// # Errors
/// [`Error::Description`] for any string outside the supported forms.
pub fn parse_description(description: &str) -> Result<Description> {
    let tokens: Vec<&str> = description.split(',').map(str::trim).collect();
    let (id_map, rest) = match tokens.split_first() {
        Some((&"IDMap", rest)) => (true, rest),
        _ => (false, tokens.as_slice()),
    };
    let kind = match rest {
        ["Flat"] => Kind::Flat,
        [h] | [h, "Flat"] if h.starts_with("HNSW") => Kind::Hnsw {
            m: number(description, h, "HNSW")?,
        },
        [i, "Flat"] if i.starts_with("IVF") => Kind::Ivf {
            nlist: number(description, i, "IVF")?,
        },
        [] | [""] => return refuse(description, format!("it is empty; supported: {SUPPORTED}")),
        other => {
            return refuse(
                description,
                format!("[{}] is not built; supported: {SUPPORTED}", other.join(",")),
            )
        }
    };
    Ok(Description { id_map, kind })
}

/// The index parameters this port applies (`ParameterSpace::set_index_parameters`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexParams {
    /// `efConstruction`.
    pub ef_construction: Option<usize>,
    /// `efSearch`.
    pub ef_search: Option<usize>,
    /// `nprobe`.
    pub nprobe: Option<usize>,
}

/// Parses `key=value,key=value` index parameters.
///
/// # Errors
/// [`Error::Invalid`] for a malformed pair or a parameter this port does not
/// apply.
pub fn parse_params(params: &str) -> Result<IndexParams> {
    let mut out = IndexParams::default();
    for pair in params.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let Some((key, value)) = pair.split_once('=') else {
            return Err(Error::Invalid(format!(
                "index parameter [{pair}] is not key=value"
            )));
        };
        let Ok(n) = value.trim().parse::<usize>() else {
            return Err(Error::Invalid(format!(
                "index parameter [{pair}] needs a whole number"
            )));
        };
        match key.trim() {
            "efConstruction" => out.ef_construction = Some(n),
            "efSearch" => out.ef_search = Some(n),
            "nprobe" => out.nprobe = Some(n),
            other => {
                return Err(Error::Invalid(format!(
                    "index parameter [{other}] is not supported"
                )))
            }
        }
    }
    Ok(out)
}
