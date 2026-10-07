//! `org.apache.lucene.analysis.cn.smart.hhmm.SegTokenFilter`: normalises a
//! segment's text -- full-width to half-width, Basic Latin to lower case,
//! every delimiter to `,`.

use super::seg_token::SegToken;
use crate::utility::{word_type, COMMON_DELIMITER};

/// `SegTokenFilter.filter(token)`.
pub fn filter(mut token: SegToken) -> SegToken {
    let lower = |c: &mut u16| {
        if (0x41..=0x5A).contains(c) {
            // ARITH: c <= 0x5A.
            #[allow(clippy::arithmetic_side_effects)]
            {
                *c += 0x20;
            }
        }
    };
    match token.word_type {
        word_type::FULLWIDTH_NUMBER | word_type::FULLWIDTH_STRING => {
            // first convert full-width -> half-width
            for c in &mut token.char_array {
                if *c >= 0xFF10 {
                    // ARITH: c >= 0xFF10 > 0xFEE0.
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        *c -= 0xFEE0;
                    }
                }
                lower(c);
            }
        }
        word_type::STRING => token.char_array.iter_mut().for_each(lower),
        word_type::DELIMITER => token.char_array = COMMON_DELIMITER.to_vec(),
        _ => {}
    }
    token
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(s: &str, t: i32) -> String {
        let u: Vec<u16> = s.encode_utf16().collect();
        String::from_utf16(&filter(SegToken::new(&u, 0, 1, t, 0)).char_array).unwrap()
    }

    #[test]
    fn normalises_as_java() {
        assert_eq!(run("ＡｂＣ１２", word_type::FULLWIDTH_STRING), "abc12");
        assert_eq!(run("１２Ａ", word_type::FULLWIDTH_NUMBER), "12a");
        assert_eq!(run("AbZ[", word_type::STRING), "abz[");
        assert_eq!(run("。", word_type::DELIMITER), ",");
        assert_eq!(run("中国", word_type::CHINESE_WORD), "中国");
    }
}
