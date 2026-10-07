//! `ko.dict.TokenInfoDictionary` and `ko.dict.TokenInfoMorphData`: the
//! system dictionary -- its FST of surface forms, target map, entry buffer
//! and part-of-speech table.
//!
//! An entry (big-endian, at its word id): `short leftId << 2 | posType`,
//! `short rightId << 2 | flags` (`HAS_SINGLE_POS` 1, `HAS_READING` 2),
//! `short wordCost`; then, without a single POS, the right POS tag byte; a
//! reading (`byte length`, then `length` UTF-16 units); or, for a
//! non-morpheme, the morphemes: `byte count`, then per morpheme its tag
//! (without a single POS) and either a string (inflected) or the length of
//! its slice of the surface. `$posDict.dat`: a tag ordinal per left id.
//!
//! Differs: every entry the target map names is walked when the dictionary
//! is read, and one that runs past the buffer, or names a tag or left id
//! outside the tables, is refused (`CorruptIndexException`) -- Java reads
//! it and fails at first use. The accessors stay bounds-checked.

use std::path::Path;
use std::sync::{Arc, LazyLock};

use lucene_analysis::morph::binary_dictionary::{
    BinaryDictionary, DICT_FILENAME_SUFFIX, POSDICT_FILENAME_SUFFIX, TARGETMAP_FILENAME_SUFFIX,
};
use lucene_analysis::morph::resource::{io_error, ResourceInput};
use lucene_analysis::morph::{MorphData, TokenInfoFst};
use lucene_analysis::AnalysisError;

use super::{inflate, read_file, Morpheme, DICT_HEADER, POSDICT_HEADER, TARGETMAP_HEADER, VERSION};
use crate::pos::{Tag, Type};

/// `TokenInfoDictionary.FST_FILENAME_SUFFIX`.
pub const FST_FILENAME_SUFFIX: &str = "$fst.dat";
/// `TokenInfoMorphData.HAS_SINGLE_POS`.
pub const HAS_SINGLE_POS: u16 = 1;
/// `TokenInfoMorphData.HAS_READING`.
pub const HAS_READING: u16 = 2;

/// `TokenInfoMorphData` (and, read through [`super::KoDict`], the
/// `UnknownMorphData` that extends it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenInfoMorphData {
    dict: BinaryDictionary,
    pos_dict: Vec<Tag>,
}

fn at(i: i32, delta: usize) -> Option<usize> {
    usize::try_from(i).ok()?.checked_add(delta)
}

impl TokenInfoMorphData {
    /// `new TokenInfoMorphData(buffer, posResource)`, then every entry of
    /// the target map checked.
    pub(crate) fn read(dict: BinaryDictionary, pos_bytes: &[u8]) -> Result<Self, AnalysisError> {
        let mut input = ResourceInput::new(pos_bytes);
        input.check_header(POSDICT_HEADER, VERSION, VERSION)?;
        let pos_size = input.read_vint()?;
        let pos_size = usize::try_from(pos_size)
            .ok()
            .filter(|&n| n <= input.remaining())
            .ok_or_else(|| io_error("EOFException", "read past EOF"))?;
        let mut pos_dict = Vec::with_capacity(pos_size);
        for _ in 0..pos_size {
            let b = input.read_byte()?;
            pos_dict.push(
                Tag::resolve(b)
                    .ok_or_else(|| io_error("ArrayIndexOutOfBoundsException", b as i8))?,
            );
        }
        let data = TokenInfoMorphData { dict, pos_dict };
        for &id in data.dict.word_ids() {
            data.check_entry(id)?;
        }
        Ok(data)
    }

    /// Walks the entry at `id` with checked reads.
    fn check_entry(&self, id: i32) -> Result<(), AnalysisError> {
        let corrupt = || {
            io_error(
                "CorruptIndexException",
                format!("dictionary entry {id} is outside the buffer"),
            )
        };
        let buf = self.dict.buffer();
        let get = |o: Option<usize>| o.and_then(|o| buf.get(o)).copied().ok_or_else(corrupt);
        get(at(id, 5))?;
        let tag = |b: u8| Tag::resolve(b).map(|_| ()).ok_or_else(corrupt);
        if self.left_pos(id).is_none() {
            return Err(corrupt());
        }
        let single = self.short(id, 2) & HAS_SINGLE_POS != 0;
        let mut offset = at(id, 6).ok_or_else(corrupt)?;
        if !single {
            tag(get(Some(offset))?)?;
        }
        if self.short(id, 2) & HAS_READING != 0 {
            // the reading at +6
            let len = usize::try_from(get(Some(offset))? as i8).map_err(|_| corrupt())?;
            get(offset.checked_add(len.saturating_mul(2)))?;
        }
        if self.pos_type(id) != Type::Morpheme {
            if !single {
                offset = offset.checked_add(1).ok_or_else(corrupt)?;
            }
            let count = u8::try_from(get(Some(offset))? as i8).map_err(|_| corrupt())?;
            offset = offset.checked_add(1).ok_or_else(corrupt)?;
            for _ in 0..count {
                if !single {
                    tag(get(Some(offset))?)?;
                    offset = offset.checked_add(1).ok_or_else(corrupt)?;
                }
                if self.pos_type(id) == Type::Inflect {
                    let len = usize::try_from(get(Some(offset))? as i8).map_err(|_| corrupt())?;
                    get(offset.checked_add(len.saturating_mul(2)))?;
                    offset = offset
                        .checked_add(len.saturating_mul(2).saturating_add(1))
                        .ok_or_else(corrupt)?;
                } else {
                    get(Some(offset))?;
                    offset = offset.checked_add(1).ok_or_else(corrupt)?;
                }
            }
        }
        Ok(())
    }

    fn byte(&self, o: Option<usize>) -> u8 {
        o.and_then(|o| self.dict.buffer().get(o))
            .copied()
            .unwrap_or(0)
    }

    fn short(&self, id: i32, delta: usize) -> u16 {
        let o = at(id, delta);
        u16::from_be_bytes([self.byte(o), self.byte(o.and_then(|o| o.checked_add(1)))])
    }

    /// The target map and buffer.
    pub fn binary_dictionary(&self) -> &BinaryDictionary {
        &self.dict
    }

    /// `getPOSType(morphId)`.
    pub fn pos_type(&self, id: i32) -> Type {
        Type::resolve((self.short(id, 0) & 3) as u8)
    }

    /// `getLeftPOS(morphId)`.
    pub fn left_pos(&self, id: i32) -> Option<Tag> {
        self.pos_dict
            .get(usize::try_from(self.left_id(id)).ok()?)
            .copied()
    }

    fn has_single_pos(&self, id: i32) -> bool {
        self.short(id, 2) & HAS_SINGLE_POS != 0
    }

    /// `getRightPOS(morphId)`.
    pub fn right_pos(&self, id: i32) -> Option<Tag> {
        let t = self.pos_type(id);
        if t == Type::Morpheme || t == Type::Compound || self.has_single_pos(id) {
            self.left_pos(id)
        } else {
            Tag::resolve(self.byte(at(id, 6)))
        }
    }

    /// `readString(offset)`: a signed length byte, then that many UTF-16
    /// units.
    fn read_string(&self, offset: usize) -> String {
        let len = usize::try_from(self.byte(Some(offset)) as i8).unwrap_or(0);
        let start = offset.saturating_add(1);
        let units: Vec<u16> = (0..len)
            .map(|i| {
                let o = start.saturating_add(i.saturating_mul(2));
                u16::from_be_bytes([self.byte(Some(o)), self.byte(o.checked_add(1))])
            })
            .collect();
        String::from_utf16_lossy(&units)
    }

    /// `getReading(morphId)`.
    pub fn reading(&self, id: i32) -> Option<String> {
        (self.short(id, 2) & HAS_READING != 0)
            .then(|| self.read_string(at(id, 6).unwrap_or(usize::MAX)))
    }

    /// `getMorphemes(morphId, surfaceForm, off, len)`: `None` for a
    /// morpheme (or an entry of no morphemes).
    pub fn morphemes(
        &self,
        id: i32,
        surface: &[u16],
        off: i32,
        _len: i32,
    ) -> Option<Vec<Morpheme>> {
        let pos_type = self.pos_type(id);
        if pos_type == Type::Morpheme {
            return None;
        }
        let mut offset = at(id, 6)?;
        let single = self.has_single_pos(id);
        if !single {
            offset = offset.checked_add(1)?; // skip rightPOS
        }
        let length = self.byte(Some(offset)) as i8;
        offset = offset.checked_add(1)?;
        if length == 0 {
            return None;
        }
        let left_pos = self.left_pos(id)?;
        let mut surface_offset = usize::try_from(off).ok()?;
        // ALLOC: `length` is one signed byte, at most 127.
        let mut out = Vec::with_capacity(usize::try_from(length).unwrap_or(0));
        for _ in 0..length.max(0) {
            let tag = if single {
                left_pos
            } else {
                let t = Tag::resolve(self.byte(Some(offset)))?;
                offset = offset.checked_add(1)?;
                t
            };
            let form = if pos_type == Type::Inflect {
                let s = self.read_string(offset);
                offset = offset
                    .checked_add(s.encode_utf16().count().saturating_mul(2).saturating_add(1))?;
                s
            } else {
                let form_len = usize::try_from(self.byte(Some(offset)) as i8).unwrap_or(0);
                offset = offset.checked_add(1)?;
                let end = surface_offset.checked_add(form_len)?;
                let s = String::from_utf16_lossy(surface.get(surface_offset..end)?);
                surface_offset = end;
                s
            };
            out.push(Morpheme {
                pos_tag: tag,
                surface_form: form,
            });
        }
        Some(out)
    }
}

impl MorphData for TokenInfoMorphData {
    /// `buffer.getShort(morphId) >>> 2` (a signed short, as Java widens it).
    #[inline]
    fn left_id(&self, id: i32) -> i32 {
        ((i32::from(self.short(id, 0) as i16) as u32) >> 2) as i32
    }
    #[inline]
    fn right_id(&self, id: i32) -> i32 {
        ((i32::from(self.short(id, 2) as i16) as u32) >> 2) as i32
    }
    #[inline]
    fn word_cost(&self, id: i32) -> i32 {
        i32::from(self.short(id, 4) as i16)
    }
}

/// `TokenInfoDictionary`.
#[derive(Debug)]
pub struct TokenInfoDictionary {
    morph_atts: TokenInfoMorphData,
    fst: Arc<TokenInfoFst>,
}

impl TokenInfoDictionary {
    /// The dictionary over a target map, part-of-speech table, entry buffer
    /// and FST, each a file's bytes.
    pub fn read(
        target_map: &[u8],
        pos_dict: &[u8],
        dict: &[u8],
        fst: &[u8],
    ) -> Result<Self, AnalysisError> {
        let bin = BinaryDictionary::read(target_map, dict, TARGETMAP_HEADER, DICT_HEADER, VERSION)?;
        let morph_atts = TokenInfoMorphData::read(bin, pos_dict)?;
        // Root arcs of the Hangul syllables (0xAC00-0xD7A3) cached.
        let fst = Arc::new(TokenInfoFst::read(fst, 0xD7A3, 0xAC00)?);
        Ok(TokenInfoDictionary { morph_atts, fst })
    }

    /// `new TokenInfoDictionary(targetMapFile, posDictFile, dictFile,
    /// fstFile)`.
    pub fn from_paths(
        target_map: &Path,
        pos_dict: &Path,
        dict: &Path,
        fst: &Path,
    ) -> Result<Self, AnalysisError> {
        Self::read(
            &read_file(target_map)?,
            &read_file(pos_dict)?,
            &read_file(dict)?,
            &read_file(fst)?,
        )
    }

    /// The four files of a dictionary named `prefix` in `dir` (Lucene's
    /// resource names).
    pub fn from_dir(dir: &Path, prefix: &str) -> Result<Self, AnalysisError> {
        let f = |suffix: &str| dir.join(format!("{prefix}{suffix}"));
        Self::from_paths(
            &f(TARGETMAP_FILENAME_SUFFIX),
            &f(POSDICT_FILENAME_SUFFIX),
            &f(DICT_FILENAME_SUFFIX),
            &f(FST_FILENAME_SUFFIX),
        )
    }

    /// `getInstance()`: the mecab-ko-dic dictionary Lucene's jar carries.
    pub fn instance() -> Arc<TokenInfoDictionary> {
        static INSTANCE: LazyLock<Arc<TokenInfoDictionary>> = LazyLock::new(|| {
            Arc::new(
                TokenInfoDictionary::read(
                    &inflate(include_bytes!("../resources/token_info_target_map.dat.z")),
                    &inflate(include_bytes!("../resources/token_info_pos_dict.dat.z")),
                    &inflate(include_bytes!("../resources/token_info_buffer.dat.z")),
                    &inflate(include_bytes!("../resources/token_info_fst.dat.z")),
                )
                .expect("the vendored mecab-ko-dic dictionary reads"),
            )
        });
        Arc::clone(&INSTANCE)
    }

    /// `getMorphAttributes()`.
    pub fn morph_attributes(&self) -> &TokenInfoMorphData {
        &self.morph_atts
    }

    /// `getFST()`.
    pub fn fst(&self) -> &Arc<TokenInfoFst> {
        &self.fst
    }

    /// `lookupWordIds(sourceId, ref)`.
    pub fn lookup_word_ids(&self, source_id: i32) -> &[i32] {
        self.morph_atts.dict.lookup_word_ids(source_id)
    }
}
