//! The `stempel` data pack ([`lucene_analysis::data_pack`]): the module's
//! dictionary files, read from a pack directory or, with the
//! `embedded-data` feature (on by default), from the copy compiled in.

use std::borrow::Cow;

use lucene_analysis::data_pack::{self, Resource};
use lucene_analysis::factory::FactoryError;
use lucene_analysis::pack_resource;

/// The pack's name: its directory under a pack directory.
pub const NAME: &str = "stempel";

/// The pack's files.
pub static RESOURCES: [Resource; 1] = [pack_resource!("stempel", "resources/stemmer_20000.tbl.z")];

/// Every file of the pack readable, else which one is not and where it
/// was looked for. The module's factories check this first.
pub fn require() -> Result<(), FactoryError> {
    data_pack::require(&RESOURCES)
}

/// The bytes of the pack's `file`; the caller checked [`require`].
pub(crate) fn get(file: &str) -> Cow<'static, [u8]> {
    let r = RESOURCES
        .iter()
        .find(|r| r.file == file)
        .expect("a file of the pack");
    data_pack::load(r).unwrap_or_else(|e| panic!("{e}"))
}
