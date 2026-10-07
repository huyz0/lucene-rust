//! Commons Codec's `StringEncoder`s as one value: what `PhoneticFilter`
//! runs and `PhoneticFilterFactory` instantiates by class name.

use crate::bm::{NameType, PhoneticEngine, RuleType};
use crate::caverphone::{caverphone1, caverphone2};
use crate::cologne::cologne_phonetic;
use crate::daitch_mokotoff::DaitchMokotoffSoundex;
use crate::double_metaphone::DoubleMetaphone;
use crate::match_rating::match_rating_encode;
use crate::metaphone::Metaphone;
use crate::nysiis::Nysiis;
use crate::soundex::{RefinedSoundex, Soundex};
use crate::EncoderError;

/// A Commons Codec phonetic encoder (an `org.apache.commons.codec.StringEncoder`
/// of the `language` packages), with Java's default constructor state.
#[derive(Debug, Clone)]
pub enum Encoder {
    /// `Soundex`.
    Soundex(Soundex),
    /// `RefinedSoundex`.
    RefinedSoundex(RefinedSoundex),
    /// `Metaphone`.
    Metaphone(Metaphone),
    /// `DoubleMetaphone` (`encode` is the primary code).
    DoubleMetaphone(DoubleMetaphone),
    /// `Caverphone1`.
    Caverphone1,
    /// `Caverphone2`, and the deprecated `Caverphone` (which delegates to it).
    Caverphone2,
    /// `ColognePhonetic`.
    ColognePhonetic,
    /// `Nysiis`.
    Nysiis(Nysiis),
    /// `MatchRatingApproachEncoder`.
    MatchRatingApproach,
    /// `DaitchMokotoffSoundex` (`encode` is the first branch's code).
    DaitchMokotoffSoundex(DaitchMokotoffSoundex),
    /// `bm.BeiderMorseEncoder` (generic names, approximate rules, concat).
    BeiderMorse(PhoneticEngine),
}

/// The package `PhoneticFilterFactory` resolves a short class name in.
pub const LANGUAGE_PACKAGE: &str = "org.apache.commons.codec.language.";

impl Encoder {
    /// `Class.forName(name).getConstructor().newInstance()` for the
    /// encoders of Commons Codec's `language` and `language.bm` packages;
    /// `None` for any other class (see the crate's parity row: the
    /// non-phonetic `Encoder`s of Commons Codec, such as `Hex` or
    /// `URLCodec`, are not ported).
    pub fn for_class_name(class_name: &str) -> Option<Encoder> {
        let simple = class_name.strip_prefix(LANGUAGE_PACKAGE)?;
        Some(match simple {
            "Soundex" => Encoder::Soundex(Soundex::default()),
            "RefinedSoundex" => Encoder::RefinedSoundex(RefinedSoundex::default()),
            "Metaphone" => Encoder::Metaphone(Metaphone::default()),
            "DoubleMetaphone" => Encoder::DoubleMetaphone(DoubleMetaphone::default()),
            "Caverphone1" => Encoder::Caverphone1,
            "Caverphone2" | "Caverphone" => Encoder::Caverphone2,
            "ColognePhonetic" => Encoder::ColognePhonetic,
            "Nysiis" => Encoder::Nysiis(Nysiis::default()),
            "MatchRatingApproachEncoder" => Encoder::MatchRatingApproach,
            "DaitchMokotoffSoundex" => {
                Encoder::DaitchMokotoffSoundex(DaitchMokotoffSoundex::default())
            }
            "bm.BeiderMorseEncoder" => Encoder::BeiderMorse(
                PhoneticEngine::new(NameType::Generic, RuleType::Approx, true)
                    .expect("APPROX is an engine rule type"),
            ),
            _ => return None,
        })
    }

    /// The Java class name.
    pub fn class_name(&self) -> &'static str {
        match self {
            Encoder::Soundex(_) => "org.apache.commons.codec.language.Soundex",
            Encoder::RefinedSoundex(_) => "org.apache.commons.codec.language.RefinedSoundex",
            Encoder::Metaphone(_) => "org.apache.commons.codec.language.Metaphone",
            Encoder::DoubleMetaphone(_) => "org.apache.commons.codec.language.DoubleMetaphone",
            Encoder::Caverphone1 => "org.apache.commons.codec.language.Caverphone1",
            Encoder::Caverphone2 => "org.apache.commons.codec.language.Caverphone2",
            Encoder::ColognePhonetic => "org.apache.commons.codec.language.ColognePhonetic",
            Encoder::Nysiis(_) => "org.apache.commons.codec.language.Nysiis",
            Encoder::MatchRatingApproach => {
                "org.apache.commons.codec.language.MatchRatingApproachEncoder"
            }
            Encoder::DaitchMokotoffSoundex(_) => {
                "org.apache.commons.codec.language.DaitchMokotoffSoundex"
            }
            Encoder::BeiderMorse(_) => "org.apache.commons.codec.language.bm.BeiderMorseEncoder",
        }
    }

    /// Whether the class has `setMaxCodeLen(int)` (`Metaphone`,
    /// `DoubleMetaphone`).
    pub fn supports_max_code_len(&self) -> bool {
        matches!(self, Encoder::Metaphone(_) | Encoder::DoubleMetaphone(_))
    }

    /// `setMaxCodeLen(int)`; a no-op for an encoder without it.
    pub fn set_max_code_len(&mut self, max_code_len: i32) {
        match self {
            Encoder::Metaphone(m) => m.set_max_code_len(max_code_len),
            Encoder::DoubleMetaphone(d) => d.set_max_code_len(max_code_len),
            _ => {}
        }
    }

    /// `encoder.encode(value).toString()` as `PhoneticFilter` calls it: the
    /// code, or the exception Java throws (an encoder answering `null`
    /// makes `toString()` throw [`EncoderError::null_result`]).
    pub fn encode(&self, value: &[u16]) -> Result<Vec<u16>, EncoderError> {
        Ok(match self {
            Encoder::Soundex(s) => s.soundex(value)?,
            Encoder::RefinedSoundex(r) => r.soundex(value),
            Encoder::Metaphone(m) => m.metaphone(value),
            Encoder::DoubleMetaphone(d) => d
                .double_metaphone(value, false)
                .ok_or_else(EncoderError::null_result)?,
            Encoder::Caverphone1 => caverphone1(value),
            Encoder::Caverphone2 => caverphone2(value),
            Encoder::ColognePhonetic => cologne_phonetic(value),
            Encoder::Nysiis(n) => n.nysiis(value),
            Encoder::MatchRatingApproach => match_rating_encode(value),
            Encoder::DaitchMokotoffSoundex(d) => d.encode(value),
            Encoder::BeiderMorse(e) => e.encode(value)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::{string, units};

    #[test]
    fn every_class_resolves_and_encodes() {
        for simple in [
            "Soundex",
            "RefinedSoundex",
            "Metaphone",
            "DoubleMetaphone",
            "Caverphone1",
            "Caverphone2",
            "Caverphone",
            "ColognePhonetic",
            "Nysiis",
            "MatchRatingApproachEncoder",
            "DaitchMokotoffSoundex",
            "bm.BeiderMorseEncoder",
        ] {
            let name = format!("{LANGUAGE_PACKAGE}{simple}");
            let mut e = Encoder::for_class_name(&name).unwrap();
            assert!(e.class_name().ends_with(
                simple
                    .trim_start_matches("bm.")
                    .trim_end_matches("Caverphone")
            ));
            assert!(!e.encode(&units("Smith")).unwrap().is_empty(), "{simple}");
            let supports = e.supports_max_code_len();
            e.set_max_code_len(1);
            if supports {
                assert_eq!(e.encode(&units("Smith")).unwrap().len(), 1);
            }
        }
        assert!(Encoder::for_class_name("org.apache.commons.codec.binary.Hex").is_none());
        assert!(Encoder::for_class_name("Soundex").is_none());
        let dm =
            Encoder::for_class_name("org.apache.commons.codec.language.DoubleMetaphone").unwrap();
        assert_eq!(
            dm.encode(&units(" ")).unwrap_err().java_class(),
            "NullPointerException"
        );
        let sx = Encoder::for_class_name("org.apache.commons.codec.language.Soundex").unwrap();
        assert!(sx.encode(&units("é")).is_err());
        assert_eq!(string(&sx.encode(&units("Smith")).unwrap()), "S530");
    }
}
