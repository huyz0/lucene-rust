//! lucene-store: Directory/IndexInput abstractions. See /PLAN.md.

pub mod codec_util;
pub mod data_input;
pub mod data_output;
pub mod directory;
pub mod error;

pub use data_input::{DataInput, SliceInput};
pub use data_output::{DataOutput, VecDataOutput};
pub use directory::{Directory, FsDirectory, Input, MmapDirectory};
pub use error::{Error, Result};
