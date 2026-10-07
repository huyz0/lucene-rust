//! `ko.dict.ConnectionCosts`: mecab-ko-dic's bigram costs over
//! [`lucene_analysis::morph::ConnectionCosts`].

use std::path::Path;
use std::sync::{Arc, LazyLock};

use lucene_analysis::morph::ConnectionCosts as MorphConnectionCosts;
use lucene_analysis::AnalysisError;

use super::{inflate, read_file, CONN_COSTS_HEADER, VERSION};

/// `ko.dict.ConnectionCosts`.
pub struct ConnectionCosts;

impl ConnectionCosts {
    /// The costs in a `ConnectionCosts.dat` file's bytes.
    pub fn read(bytes: &[u8]) -> Result<MorphConnectionCosts, AnalysisError> {
        MorphConnectionCosts::read(bytes, CONN_COSTS_HEADER, VERSION)
    }

    /// `new ConnectionCosts(Path)`.
    pub fn from_path(path: &Path) -> Result<MorphConnectionCosts, AnalysisError> {
        Self::read(&read_file(path)?)
    }

    /// `getInstance()`.
    pub fn instance() -> Arc<MorphConnectionCosts> {
        static INSTANCE: LazyLock<Arc<MorphConnectionCosts>> = LazyLock::new(|| {
            Arc::new(
                ConnectionCosts::read(&inflate(include_bytes!(
                    "../resources/connection_costs.dat.z"
                )))
                .expect("the vendored connection costs read"),
            )
        });
        Arc::clone(&INSTANCE)
    }
}
