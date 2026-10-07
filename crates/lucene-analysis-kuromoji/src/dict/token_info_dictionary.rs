//! `ja.dict.TokenInfoDictionary` and `ja.dict.TokenInfoMorphData`: the
//! system dictionary -- its FST of surface forms, target map, entry buffer
//! and part-of-speech table.
//!
//! An entry (big-endian, at its word id): `short leftId << 3 | flags`
//! (`HAS_BASEFORM` 1, `HAS_READING` 2, `HAS_PRONUNCIATION` 4; the left id
//! doubles as the right id and as the part-of-speech index), `short
//! wordCost`, then optionally a base form (`byte prefix << 4 | suffix`,
//! `suffix` UTF-16 units: the base form is the surface's first `prefix`
//! units then these), a reading and a pronunciation (each `byte length <<
//! 1 | kana`, then `length` bytes of katakana offset from U+30A0 when
//! `kana`, else `length` UTF-16 units). `$posDict.dat`: per left id three
//! strings, part of speech, inflection type and form (empty: `null`).
//!
//! Differs: every entry the target map names is walked when the dictionary
//! is read, and one that runs past the buffer, or whose left id is outside
//! the part-of-speech table, is refused (`CorruptIndexException`) -- Java
//! reads it and fails at first use (`IndexOutOfBoundsException`). The
//! accessors stay bounds-checked (a value outside the buffer reads as
//! zero) so no word id can panic them.

use std::path::Path;
use std::sync::{Arc, LazyLock};

use lucene_analysis::morph::binary_dictionary::{
    BinaryDictionary, DICT_FILENAME_SUFFIX, POSDICT_FILENAME_SUFFIX, TARGETMAP_FILENAME_SUFFIX,
};
use lucene_analysis::morph::resource::{io_error, ResourceInput};
use lucene_analysis::morph::{MorphData, TokenInfoFst};
use lucene_analysis::AnalysisError;

use super::{inflate, read_file, DICT_HEADER, POSDICT_HEADER, TARGETMAP_HEADER, VERSION};

/// `TokenInfoDictionary.FST_FILENAME_SUFFIX`.
pub const FST_FILENAME_SUFFIX: &str = "$fst.dat";
/// `TokenInfoMorphData.HAS_BASEFORM`.
pub const HAS_BASEFORM: u16 = 1;
/// `TokenInfoMorphData.HAS_READING`.
pub const HAS_READING: u16 = 2;
/// `TokenInfoMorphData.HAS_PRONUNCIATION`.
pub const HAS_PRONUNCIATION: u16 = 4;

/// `TokenInfoMorphData` (and, read through [`super::JaDict`], the
/// `UnknownMorphData` that extends it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenInfoMorphData {
    dict: BinaryDictionary,
    pos_dict: Vec<String>,
    infl_type_dict: Vec<Option<String>>,
    infl_form_dict: Vec<Option<String>>,
}

fn at(i: i32, delta: i32) -> Option<usize> {
    usize::try_from(i)
        .ok()?
        .checked_add(usize::try_from(delta).ok()?)
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
        let mut infl_type_dict = Vec::with_capacity(pos_size);
        let mut infl_form_dict = Vec::with_capacity(pos_size);
        let non_empty = |s: String| (!s.is_empty()).then_some(s);
        for _ in 0..pos_size {
            pos_dict.push(input.read_string()?);
            // this is how we encode null inflections
            infl_type_dict.push(non_empty(input.read_string()?));
            infl_form_dict.push(non_empty(input.read_string()?));
        }
        let data = TokenInfoMorphData {
            dict,
            pos_dict,
            infl_type_dict,
            infl_form_dict,
        };
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
        get(at(id, 3))?;
        if usize::try_from(self.left_id(id)).map_or(true, |l| l >= self.pos_dict.len()) {
            return Err(io_error(
                "CorruptIndexException",
                format!(
                    "dictionary entry {id}: left id {} has no part of speech",
                    self.left_id(id)
                ),
            ));
        }
        let mut offset = at(id, 4).ok_or_else(corrupt)?;
        if self.has(id, HAS_BASEFORM) {
            let suffix = usize::from(get(Some(offset))? & 0xF);
            offset = offset
                .checked_add(suffix.saturating_mul(2).saturating_add(1))
                .ok_or_else(corrupt)?;
        }
        for flag in [HAS_READING, HAS_PRONUNCIATION] {
            if self.has(id, flag) {
                let data = get(Some(offset))?;
                let len = if data & 1 == 0 {
                    usize::from(data & 0xFE)
                } else {
                    usize::from(data >> 1)
                };
                offset = offset
                    .checked_add(len.saturating_add(1))
                    .ok_or_else(corrupt)?;
            }
        }
        if offset > buf.len() {
            return Err(corrupt());
        }
        Ok(())
    }

    fn byte(&self, o: Option<usize>) -> u8 {
        o.and_then(|o| self.dict.buffer().get(o))
            .copied()
            .unwrap_or(0)
    }

    fn short(&self, id: i32, delta: i32) -> u16 {
        let o = at(id, delta);
        u16::from_be_bytes([self.byte(o), self.byte(o.and_then(|o| o.checked_add(1)))])
    }

    fn has(&self, id: i32, flag: u16) -> bool {
        self.short(id, 0) & flag != 0
    }

    /// `hasPronunciationData(wordId)`.
    pub(crate) fn has_pronunciation_data(&self, id: i32) -> bool {
        self.has(id, HAS_PRONUNCIATION)
    }

    /// The target map and buffer.
    pub fn binary_dictionary(&self) -> &BinaryDictionary {
        &self.dict
    }

    /// `getPartOfSpeech(morphId)`.
    pub fn part_of_speech(&self, id: i32) -> Option<&str> {
        self.pos_dict
            .get(usize::try_from(self.left_id(id)).ok()?)
            .map(String::as_str)
    }

    /// `getInflectionType(morphId)`.
    pub fn inflection_type(&self, id: i32) -> Option<&str> {
        self.infl_type_dict
            .get(usize::try_from(self.left_id(id)).ok()?)?
            .as_deref()
    }

    /// `getInflectionForm(wordId)`.
    pub fn inflection_form(&self, id: i32) -> Option<&str> {
        self.infl_form_dict
            .get(usize::try_from(self.left_id(id)).ok()?)?
            .as_deref()
    }

    /// `getBaseForm(morphId, surfaceForm, off, len)`.
    pub fn base_form(&self, id: i32, surface: &[u16], off: i32, _len: i32) -> Option<String> {
        if !self.has(id, HAS_BASEFORM) {
            return None;
        }
        let offset = at(id, 4)?;
        let data = self.byte(Some(offset));
        let prefix = usize::from(data >> 4);
        let suffix = usize::from(data & 0xF);
        let start = usize::try_from(off).unwrap_or(0);
        let mut text: Vec<u16> = surface.iter().skip(start).take(prefix).copied().collect();
        for i in 0..suffix {
            let o = offset.checked_add(1)?.checked_add(i.checked_mul(2)?)?;
            text.push(u16::from_be_bytes([
                self.byte(Some(o)),
                self.byte(o.checked_add(1)),
            ]));
        }
        Some(String::from_utf16_lossy(&text))
    }

    /// `readingOffset(wordId)`.
    fn reading_offset(&self, id: i32) -> Option<usize> {
        let offset = at(id, 4)?;
        if self.has(id, HAS_BASEFORM) {
            let len = usize::from(self.byte(Some(offset)) & 0xF);
            offset.checked_add(1)?.checked_add(len.checked_mul(2)?)
        } else {
            Some(offset)
        }
    }

    /// `pronunciationOffset(wordId)`.
    fn pronunciation_offset(&self, id: i32) -> Option<usize> {
        let offset = self.reading_offset(id)?;
        if self.has(id, HAS_READING) {
            let data = self.byte(Some(offset));
            let len = if data & 1 == 0 {
                usize::from(data & 0xFE) // UTF-16: mask off kana bit
            } else {
                usize::from(data >> 1)
            };
            offset.checked_add(1)?.checked_add(len)
        } else {
            Some(offset)
        }
    }

    /// `readString(offset, length, kana)` of the `byte length << 1 | kana`
    /// header at `offset`.
    fn read_string(&self, offset: usize) -> String {
        let data = self.byte(Some(offset));
        let length = usize::from(data >> 1);
        let start = offset.saturating_add(1);
        let units: Vec<u16> = if data & 1 == 1 {
            (0..length)
                .map(|i| 0x30A0u16.wrapping_add(u16::from(self.byte(start.checked_add(i)))))
                .collect()
        } else {
            (0..length)
                .map(|i| {
                    let o = start.saturating_add(i.saturating_mul(2));
                    u16::from_be_bytes([self.byte(Some(o)), self.byte(o.checked_add(1))])
                })
                .collect()
        };
        String::from_utf16_lossy(&units)
    }

    /// `getReading(morphId, surface, off, len)`: the stored reading, else
    /// the surface with hiragana shifted to katakana.
    pub fn reading(&self, id: i32, surface: &[u16], off: i32, len: i32) -> String {
        if self.has(id, HAS_READING) {
            if let Some(o) = self.reading_offset(id) {
                return self.read_string(o);
            }
        }
        let start = usize::try_from(off).unwrap_or(0);
        let len = usize::try_from(len).unwrap_or(0);
        let text: Vec<u16> = surface
            .iter()
            .skip(start)
            .take(len)
            .map(|&ch| {
                if ch > 0x3040 && ch < 0x3097 {
                    ch.wrapping_add(0x60)
                } else {
                    ch
                }
            })
            .collect();
        String::from_utf16_lossy(&text)
    }

    /// `getPronunciation(morphId, surface, off, len)`: the stored
    /// pronunciation, else the reading.
    pub fn pronunciation(
        &self,
        id: i32,
        surface: &[u16],
        off: i32,
        len: i32,
        _unknown: bool,
    ) -> String {
        if self.has(id, HAS_PRONUNCIATION) {
            if let Some(o) = self.pronunciation_offset(id) {
                return self.read_string(o);
            }
        }
        self.reading(id, surface, off, len)
    }
}

impl MorphData for TokenInfoMorphData {
    #[inline]
    fn left_id(&self, id: i32) -> i32 {
        i32::from(self.short(id, 0) >> 3)
    }
    #[inline]
    fn right_id(&self, id: i32) -> i32 {
        i32::from(self.short(id, 0) >> 3)
    }
    #[inline]
    fn word_cost(&self, id: i32) -> i32 {
        i32::from(self.short(id, 2) as i16)
    }
    #[inline]
    fn connection(&self, id: i32) -> (i32, i32, i32) {
        // The entry's two shorts in one read (an entry is never at the
        // buffer's very end; the byte-wise reads cover one that is).
        if let Some(&[a, b, c, d]) =
            at(id, 0).and_then(|o| self.dict.buffer().get(o..o.checked_add(4)?))
        {
            let ids = i32::from(u16::from_be_bytes([a, b]) >> 3);
            return (ids, ids, i32::from(i16::from_be_bytes([c, d])));
        }
        let ids = i32::from(self.short(id, 0) >> 3);
        (ids, ids, self.word_cost(id))
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
    /// and FST, each a file's bytes (`new TokenInfoDictionary(...)` over
    /// resources).
    pub fn read(
        target_map: &[u8],
        pos_dict: &[u8],
        dict: &[u8],
        fst: &[u8],
    ) -> Result<Self, AnalysisError> {
        let bin = BinaryDictionary::read(target_map, dict, TARGETMAP_HEADER, DICT_HEADER, VERSION)?;
        let morph_atts = TokenInfoMorphData::read(bin, pos_dict)?;
        // fasterButMoreRam: kana + han (0x3040-0x9FFF) root arcs cached.
        let fst = Arc::new(TokenInfoFst::read(fst, 0x9FFF, 0x3040)?);
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

    /// The four files of a dictionary named `prefix` in `dir`
    /// (`<prefix>$targetMap.dat`, ...: Lucene's resource names).
    pub fn from_dir(dir: &Path, prefix: &str) -> Result<Self, AnalysisError> {
        let f = |suffix: &str| dir.join(format!("{prefix}{suffix}"));
        Self::from_paths(
            &f(TARGETMAP_FILENAME_SUFFIX),
            &f(POSDICT_FILENAME_SUFFIX),
            &f(DICT_FILENAME_SUFFIX),
            &f(FST_FILENAME_SUFFIX),
        )
    }

    /// `getInstance()`: the IPADIC dictionary Lucene's jar carries.
    pub fn instance() -> Arc<TokenInfoDictionary> {
        static INSTANCE: LazyLock<Arc<TokenInfoDictionary>> = LazyLock::new(|| {
            Arc::new(
                TokenInfoDictionary::read(
                    &inflate(include_bytes!("../resources/token_info_target_map.dat.z")),
                    &inflate(include_bytes!("../resources/token_info_pos_dict.dat.z")),
                    &inflate(include_bytes!("../resources/token_info_buffer.dat.z")),
                    &inflate(include_bytes!("../resources/token_info_fst.dat.z")),
                )
                .expect("the vendored IPADIC dictionary reads"),
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
