//! `readPlanetObject` over arbitrary bytes: the geo3d deserialization a
//! serialized shape doc value (`Geo3dBinaryCodec`) goes through. Every input
//! must come back as a value or an `Err` -- a panic, a stack overflow from
//! nesting (the 64-level limit) or an allocation sized by a corrupt count is
//! the finding. A shape that does read is written back, and must re-read to
//! the same bytes.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lucene_util::spatial3d::serializable::Input;
use lucene_util::spatial3d::standard_objects::{read_planet_object, write_planet_object};

fuzz_target!(|data: &[u8]| {
    let Ok(object) = read_planet_object(&mut Input::new(data)) else {
        return;
    };
    let planet = object
        .as_planet_object()
        .expect("read_planet_object returned a non-planet object");
    let mut once = Vec::new();
    if write_planet_object(&mut once, &*planet).is_err() {
        return;
    }
    let again = read_planet_object(&mut Input::new(&once)).expect("a written shape re-reads");
    let mut twice = Vec::new();
    write_planet_object(&mut twice, &*again.as_planet_object().unwrap()).expect("re-writes");
    assert_eq!(once, twice, "write -> read -> write is stable");
});
