//! `org.apache.lucene.analysis.morph.CharacterDefinition`: the character
//! class of every UTF-16 unit, and per class whether unknown words are
//! always proposed (`invoke`) and whether a run of the class is one unknown
//! word (`group`).
//!
//! File: `CodecUtil` header, 65,536 class bytes (one per unit), then one
//! byte per class: bit 0 `invoke`, bit 1 `group`.
//!
//! Differs: Java reads a class byte outside `0..classCount` and fails with
//! `ArrayIndexOutOfBoundsException` at the first `isInvoke`/`isGroup` of
//! such a unit; this refuses the file when it is read
//! (`CorruptIndexException`).

use super::resource::{io_error, ResourceInput};
use crate::AnalysisError;

/// The number of UTF-16 units (`characterCategoryMap.length`).
const UNITS: usize = 0x10000;

/// `CharacterDefinition`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacterDefinition {
    character_category_map: Vec<u8>,
    invoke_map: Vec<bool>,
    group_map: Vec<bool>,
}

impl CharacterDefinition {
    /// `new CharacterDefinition(resource, codecHeader, version, classCount)`.
    pub fn read(
        bytes: &[u8],
        codec_header: &str,
        version: i32,
        class_count: usize,
    ) -> Result<Self, AnalysisError> {
        let mut input = ResourceInput::new(bytes);
        input.check_header(codec_header, version, version)?;
        let character_category_map = input.read_bytes(UNITS)?.to_vec();
        let mut invoke_map = Vec::with_capacity(class_count);
        let mut group_map = Vec::with_capacity(class_count);
        for _ in 0..class_count {
            let b = input.read_byte()?;
            invoke_map.push(b & 0x01 != 0);
            group_map.push(b & 0x02 != 0);
        }
        if let Some((unit, &class)) = character_category_map
            .iter()
            .enumerate()
            .find(|(_, &c)| usize::from(c) >= class_count)
        {
            return Err(io_error(
                "CorruptIndexException",
                format!(
                    "character class {} of U+{unit:04X} is not below {class_count}",
                    class as i8
                ),
            ));
        }
        Ok(CharacterDefinition {
            character_category_map,
            invoke_map,
            group_map,
        })
    }

    /// `getCharacterClass(char)`.
    #[inline]
    pub fn character_class(&self, c: u16) -> u8 {
        self.character_category_map
            .get(usize::from(c))
            .copied()
            .unwrap_or(0)
    }

    /// `isInvoke(char)`.
    #[inline]
    pub fn is_invoke(&self, c: u16) -> bool {
        self.invoke_map
            .get(usize::from(self.character_class(c)))
            .copied()
            .unwrap_or(false)
    }

    /// `isGroup(char)`.
    #[inline]
    pub fn is_group(&self, c: u16) -> bool {
        self.group_map
            .get(usize::from(self.character_class(c)))
            .copied()
            .unwrap_or(false)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::morph::resource::test_util::header;

    /// A definition file: every unit in class `fill` except `classes`;
    /// `flags` per class.
    pub(crate) fn definition_file(
        codec: &str,
        fill: u8,
        classes: &[(u16, u8)],
        flags: &[u8],
    ) -> Vec<u8> {
        let mut map = vec![fill; UNITS];
        for &(u, c) in classes {
            map[usize::from(u)] = c;
        }
        let mut b = header(codec, 1);
        b.extend_from_slice(&map);
        b.extend_from_slice(flags);
        b
    }

    #[test]
    fn classes_and_flags() {
        let f = definition_file("ja_cd", 1, &[(0x3042, 2), (b'a'.into(), 0)], &[0, 1, 2]);
        let d = CharacterDefinition::read(&f, "ja_cd", 1, 3).unwrap();
        assert_eq!(d.character_class(0x3042), 2);
        assert_eq!(d.character_class(u16::from(b'a')), 0);
        assert_eq!(d.character_class(0xFFFF), 1);
        assert!(d.is_invoke(0x20) && !d.is_group(0x20));
        assert!(!d.is_invoke(0x3042) && d.is_group(0x3042));
        assert!(!d.is_invoke(u16::from(b'a')) && !d.is_group(u16::from(b'a')));
    }

    #[test]
    fn corrupt_definitions_fail() {
        let bad = definition_file("ja_cd", 1, &[(7, 0xFF)], &[0, 0]);
        let e = CharacterDefinition::read(&bad, "ja_cd", 1, 2).unwrap_err();
        assert!(e.to_string().contains("CorruptIndexException"), "{e}");
        let f = definition_file("ja_cd", 0, &[], &[0]);
        for cut in [0, 10, f.len() - 1] {
            assert!(CharacterDefinition::read(&f[..cut], "ja_cd", 1, 1).is_err());
        }
    }
}
