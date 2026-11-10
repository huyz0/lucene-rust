//! Painless executed natively: the IR Painless's own front end produces
//! (types, casts, methods and `def` sites resolved) compiled to a register
//! program and run with Java's semantics.

pub mod api;
pub mod compare;
pub mod compile;
pub mod def;
pub mod ir;
pub mod jmap;
pub mod lambda;
pub mod number;
pub mod object;
pub mod types;
pub mod value;
pub mod vm;

/// Why a script cannot be loaded.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A malformed IR blob.
    #[error("malformed painless IR: {0}")]
    Ir(String),
    /// A construct or API the native runtime does not implement.
    #[error("unsupported painless construct: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, Error>;
