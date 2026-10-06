//! `org.apache.lucene.analysis.in`: `IndicNormalizer` -- composes the
//! decomposed vowel forms of the nine Indic blocks Lucene knows (Devanagari
//! .. Malayalam) into their precomposed letters. (The module is `in_`,
//! `in` being a Rust keyword.)

use crate::util::stemmer_util::delete;

use super::{CharStemmer, NormalizeFilter};

/// `scripts`: the block's flag and base, by block index from U+0900, each
/// block 0x80 wide (`Character.UnicodeBlock.of` over these ranges).
const SCRIPTS: [(i32, u16); 9] = [
    (1, 0x0900),
    (2, 0x0980),
    (4, 0x0A00),
    (8, 0x0A80),
    (16, 0x0B00),
    (32, 0x0B80),
    (64, 0x0C00),
    (128, 0x0C80),
    (256, 0x0D00),
];

/// `Character.UnicodeBlock.of(ch)`, as an index into [`SCRIPTS`] when it is
/// one of them.
fn block(ch: u16) -> Option<usize> {
    (0x0900..0x0D80)
        .contains(&ch)
        .then(|| usize::from((ch - 0x0900) >> 7))
}

/// `decompositions`: `{first, second, third or -1 (0xFF: ZWJ), composed,
/// script flags}`, relative to the block's base (generated from Lucene's
/// table).
const DECOMPOSITIONS: [[i32; 5]; 72] = [
    [0x05, 0x3e, 0x45, 0x11, 9],
    [0x05, 0x3e, 0x46, 0x12, 1],
    [0x05, 0x3e, 0x47, 0x13, 9],
    [0x05, 0x3e, 0x48, 0x14, 9],
    [0x05, 0x3e, -1, 0x06, 31],
    [0x05, 0x45, -1, 0x72, 1],
    [0x05, 0x45, -1, 0x0d, 8],
    [0x05, 0x46, -1, 0x04, 1],
    [0x05, 0x47, -1, 0x0f, 8],
    [0x05, 0x48, -1, 0x10, 12],
    [0x05, 0x49, -1, 0x11, 9],
    [0x05, 0x4a, -1, 0x12, 1],
    [0x05, 0x4b, -1, 0x13, 9],
    [0x05, 0x4c, -1, 0x14, 13],
    [0x06, 0x45, -1, 0x11, 9],
    [0x06, 0x46, -1, 0x12, 1],
    [0x06, 0x47, -1, 0x13, 9],
    [0x06, 0x48, -1, 0x14, 9],
    [0x07, 0x57, -1, 0x08, 256],
    [0x09, 0x41, -1, 0x0a, 1],
    [0x09, 0x57, -1, 0x0a, 288],
    [0x0e, 0x46, -1, 0x10, 256],
    [0x0f, 0x45, -1, 0x0d, 1],
    [0x0f, 0x46, -1, 0x0e, 1],
    [0x0f, 0x47, -1, 0x10, 1],
    [0x0f, 0x57, -1, 0x10, 16],
    [0x12, 0x3e, -1, 0x13, 256],
    [0x12, 0x4c, -1, 0x14, 192],
    [0x12, 0x55, -1, 0x13, 64],
    [0x12, 0x57, -1, 0x14, 288],
    [0x13, 0x57, -1, 0x14, 16],
    [0x15, 0x3c, -1, 0x58, 1],
    [0x16, 0x3c, -1, 0x59, 5],
    [0x17, 0x3c, -1, 0x5a, 5],
    [0x1c, 0x3c, -1, 0x5b, 5],
    [0x21, 0x3c, -1, 0x5c, 19],
    [0x22, 0x3c, -1, 0x5d, 19],
    [0x23, 0x4d, 0xff, 0x7a, 256],
    [0x24, 0x4d, 0xff, 0x4e, 2],
    [0x28, 0x3c, -1, 0x29, 1],
    [0x28, 0x4d, 0xff, 0x7b, 256],
    [0x2b, 0x3c, -1, 0x5e, 5],
    [0x2f, 0x3c, -1, 0x5f, 3],
    [0x2c, 0x41, 0x41, 0x0b, 64],
    [0x30, 0x3c, -1, 0x31, 1],
    [0x30, 0x4d, 0xff, 0x7c, 256],
    [0x32, 0x4d, 0xff, 0x7d, 256],
    [0x33, 0x3c, -1, 0x34, 1],
    [0x33, 0x4d, 0xff, 0x7e, 256],
    [0x35, 0x41, -1, 0x2e, 64],
    [0x3e, 0x45, -1, 0x49, 9],
    [0x3e, 0x46, -1, 0x4a, 1],
    [0x3e, 0x47, -1, 0x4b, 9],
    [0x3e, 0x48, -1, 0x4c, 9],
    [0x3f, 0x55, -1, 0x40, 128],
    [0x41, 0x41, -1, 0x42, 4],
    [0x46, 0x3e, -1, 0x4a, 288],
    [0x46, 0x42, 0x55, 0x4b, 128],
    [0x46, 0x42, -1, 0x4a, 128],
    [0x46, 0x46, -1, 0x48, 256],
    [0x46, 0x55, -1, 0x47, 192],
    [0x46, 0x56, -1, 0x48, 192],
    [0x46, 0x57, -1, 0x4c, 288],
    [0x47, 0x3e, -1, 0x4b, 306],
    [0x47, 0x57, -1, 0x4c, 18],
    [0x4a, 0x55, -1, 0x4b, 128],
    [0x72, 0x3f, -1, 0x07, 4],
    [0x72, 0x40, -1, 0x08, 4],
    [0x72, 0x47, -1, 0x0f, 4],
    [0x73, 0x41, -1, 0x09, 4],
    [0x73, 0x42, -1, 0x0a, 4],
    [0x73, 0x4b, -1, 0x13, 4],
];

/// `IndicNormalizer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct IndicNormalizer;

impl IndicNormalizer {
    // Java: IndicNormalizer.compose
    fn compose(ch0: i32, b0: usize, text: &mut [u16], pos: usize, mut len: usize) -> usize {
        let (flag, base) = SCRIPTS[b0];
        if pos + 1 >= len {
            return len;
        }
        let ch1 = i32::from(text[pos + 1]) - i32::from(base);
        if block(text[pos + 1]) != Some(b0) {
            return len;
        }
        let mut ch2 = -1;
        if pos + 2 < len {
            ch2 = i32::from(text[pos + 2]) - i32::from(base);
            if text[pos + 2] == 0x200D {
                ch2 = 0xFF;
            } else if block(text[pos + 2]) != Some(b0) {
                ch2 = -1;
            }
        }
        for d in DECOMPOSITIONS {
            if d[0] == ch0 && d[4] & flag != 0 && d[1] == ch1 && (d[2] < 0 || d[2] == ch2) {
                text[pos] = (i32::from(base) + d[3]) as u16;
                len = delete(text, pos + 1, len);
                if d[2] >= 0 {
                    len = delete(text, pos + 1, len);
                }
                return len;
            }
        }
        len
    }
}

impl CharStemmer for IndicNormalizer {
    // Java: IndicNormalizer.normalize
    fn stem(&self, text: &mut Vec<u16>, mut len: usize) -> usize {
        let mut i = 0;
        while i < len {
            if let Some(b) = block(text[i]) {
                let (flag, base) = SCRIPTS[b];
                let ch = i32::from(text[i] - base);
                // `decompMask.get(ch)`: some decomposition of this script starts with ch.
                if DECOMPOSITIONS
                    .iter()
                    .any(|d| d[0] == ch && d[4] & flag != 0)
                {
                    len = Self::compose(ch, b, text, i, len);
                }
            }
            i += 1;
        }
        len
    }
}

/// `IndicNormalizationFilter`.
pub type IndicNormalizationFilter<I> = NormalizeFilter<I, IndicNormalizer>;
