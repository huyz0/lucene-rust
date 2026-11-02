//! The `smartcn` data pack ([`lucene_analysis::data_pack`]): the module's
//! dictionary files, read from a pack directory or, with the
//! `embedded-data` feature (on by default), from the copy compiled in.

use std::borrow::Cow;

use lucene_analysis::data_pack::{self, Resource};
use lucene_analysis::factory::FactoryError;
use lucene_analysis::pack_resource;

/// The pack's name: its directory under a pack directory.
pub const NAME: &str = "smartcn";

/// The pack's files.
pub static RESOURCES: [Resource; 2] = [
    pack_resource!("smartcn", "resources/bigramdict.mem.z"),
    pack_resource!("smartcn", "resources/coredict.mem.z"),
];

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
