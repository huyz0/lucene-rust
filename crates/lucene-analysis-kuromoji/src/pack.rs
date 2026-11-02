//! The `kuromoji` data pack ([`lucene_analysis::data_pack`]): the module's
//! dictionary files, read from a pack directory or, with the
//! `embedded-data` feature (on by default), from the copy compiled in.

use std::borrow::Cow;

use lucene_analysis::data_pack::{self, Resource};
use lucene_analysis::factory::FactoryError;
use lucene_analysis::pack_resource;

/// The pack's name: its directory under a pack directory.
pub const NAME: &str = "kuromoji";

/// The pack's files.
pub static RESOURCES: [Resource; 9] = [
    pack_resource!("kuromoji", "resources/character_definition.dat.z"),
    pack_resource!("kuromoji", "resources/connection_costs.dat.z"),
    pack_resource!("kuromoji", "resources/token_info_buffer.dat.z"),
    pack_resource!("kuromoji", "resources/token_info_fst.dat.z"),
    pack_resource!("kuromoji", "resources/token_info_pos_dict.dat.z"),
    pack_resource!("kuromoji", "resources/token_info_target_map.dat.z"),
    pack_resource!("kuromoji", "resources/unknown_buffer.dat.z"),
    pack_resource!("kuromoji", "resources/unknown_pos_dict.dat.z"),
    pack_resource!("kuromoji", "resources/unknown_target_map.dat.z"),
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
