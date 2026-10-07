//! `org.apache.lucene.analysis.ko.POS`: mecab-ko-dic's part-of-speech
//! tags and the four kinds of dictionary entry.

use lucene_analysis::AnalysisError;

/// `POS.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    /// `MORPHEME`: a simple morpheme.
    Morpheme,
    /// `COMPOUND`: a compound noun.
    Compound,
    /// `INFLECT`: an inflected token.
    Inflect,
    /// `PREANALYSIS`: a pre-analysis token.
    Preanalysis,
}

impl Type {
    /// `Type.name()`.
    pub fn name(self) -> &'static str {
        match self {
            Type::Morpheme => "MORPHEME",
            Type::Compound => "COMPOUND",
            Type::Inflect => "INFLECT",
            Type::Preanalysis => "PREANALYSIS",
        }
    }

    /// `resolveType(byte)`: `Type.values()[type]` (the two low bits of an
    /// entry, so always in range).
    pub fn resolve(t: u8) -> Type {
        match t & 3 {
            0 => Type::Morpheme,
            1 => Type::Compound,
            2 => Type::Inflect,
            _ => Type::Preanalysis,
        }
    }

    /// `resolveType(String)`: `*` is `MORPHEME`.
    pub fn resolve_name(name: &str) -> Result<Type, AnalysisError> {
        if name == "*" {
            return Ok(Type::Morpheme);
        }
        match name.to_ascii_uppercase().as_str() {
            "MORPHEME" => Ok(Type::Morpheme),
            "COMPOUND" => Ok(Type::Compound),
            "INFLECT" => Ok(Type::Inflect),
            "PREANALYSIS" => Ok(Type::Preanalysis),
            _ => Err(AnalysisError::IllegalArgument(format!(
                "No enum constant org.apache.lucene.analysis.ko.POS.Type.{}",
                name.to_ascii_uppercase()
            ))),
        }
    }
}

/// `POS.Tag`, in declaration order (a dictionary stores the ordinal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tag {
    /// `EP` (100): Pre-final ending.
    Ep,
    /// `EF` (101): Sentence-closing ending.
    Ef,
    /// `EC` (102): Connective ending.
    Ec,
    /// `ETN` (103): Nominal transformative ending.
    Etn,
    /// `ETM` (104): Adnominal form transformative ending.
    Etm,
    /// `IC` (110): Interjection.
    Ic,
    /// `JKS` (120): Subject case marker.
    Jks,
    /// `JKC` (121): Complement case marker.
    Jkc,
    /// `JKG` (122): Adnominal case marker.
    Jkg,
    /// `JKO` (123): Object case marker.
    Jko,
    /// `JKB` (124): Adverbial case marker.
    Jkb,
    /// `JKV` (125): Vocative case marker.
    Jkv,
    /// `JKQ` (126): Quotative case marker.
    Jkq,
    /// `JX` (127): Auxiliary postpositional particle.
    Jx,
    /// `JC` (128): Conjunctive postpositional particle.
    Jc,
    /// `MAG` (130): General Adverb.
    Mag,
    /// `MAJ` (131): Conjunctive adverb.
    Maj,
    /// `MM` (140): Modifier.
    Mm,
    /// `NNG` (150): General Noun.
    Nng,
    /// `NNP` (151): Proper Noun.
    Nnp,
    /// `NNB` (152): Dependent noun.
    Nnb,
    /// `NNBC` (153): Dependent noun.
    Nnbc,
    /// `NP` (154): Pronoun.
    Np,
    /// `NR` (155): Numeral.
    Nr,
    /// `SF` (160): Terminal punctuation.
    Sf,
    /// `SH` (161): Chinese Characeter.
    Sh,
    /// `SL` (162): Foreign language.
    Sl,
    /// `SN` (163): Number.
    Sn,
    /// `SP` (164): Space.
    Sp,
    /// `SSC` (165): Closing brackets.
    Ssc,
    /// `SSO` (166): Opening brackets.
    Sso,
    /// `SC` (167): Separator.
    Sc,
    /// `SY` (168): Other symbol.
    Sy,
    /// `SE` (169): Ellipsis.
    Se,
    /// `VA` (170): Adjective.
    Va,
    /// `VCN` (171): Negative designator.
    Vcn,
    /// `VCP` (172): Positive designator.
    Vcp,
    /// `VV` (173): Verb.
    Vv,
    /// `VX` (174): Auxiliary Verb or Adjective.
    Vx,
    /// `XPN` (181): Prefix.
    Xpn,
    /// `XR` (182): Root.
    Xr,
    /// `XSA` (183): Adjective Suffix.
    Xsa,
    /// `XSN` (184): Noun Suffix.
    Xsn,
    /// `XSV` (185): Verb Suffix.
    Xsv,
    /// `UNKNOWN` (999): Unknown.
    Unknown,
    /// `UNA` (-1): Unknown.
    Una,
    /// `NA` (-1): Unknown.
    Na,
    /// `VSV` (-1): Unknown.
    Vsv,
}

/// Every tag, by ordinal (`Tag.values()`).
pub const TAGS: [Tag; 48] = [
    Tag::Ep,
    Tag::Ef,
    Tag::Ec,
    Tag::Etn,
    Tag::Etm,
    Tag::Ic,
    Tag::Jks,
    Tag::Jkc,
    Tag::Jkg,
    Tag::Jko,
    Tag::Jkb,
    Tag::Jkv,
    Tag::Jkq,
    Tag::Jx,
    Tag::Jc,
    Tag::Mag,
    Tag::Maj,
    Tag::Mm,
    Tag::Nng,
    Tag::Nnp,
    Tag::Nnb,
    Tag::Nnbc,
    Tag::Np,
    Tag::Nr,
    Tag::Sf,
    Tag::Sh,
    Tag::Sl,
    Tag::Sn,
    Tag::Sp,
    Tag::Ssc,
    Tag::Sso,
    Tag::Sc,
    Tag::Sy,
    Tag::Se,
    Tag::Va,
    Tag::Vcn,
    Tag::Vcp,
    Tag::Vv,
    Tag::Vx,
    Tag::Xpn,
    Tag::Xr,
    Tag::Xsa,
    Tag::Xsn,
    Tag::Xsv,
    Tag::Unknown,
    Tag::Una,
    Tag::Na,
    Tag::Vsv,
];

/// The code `POS.Tag` gives `UNA`, `NA` and `VSV`: a tag's data, not an
/// out-of-domain sentinel (no caller branches on it).
pub const UNKNOWN_TAG_CODE: i32 = -1;

impl Tag {
    /// `name()`.
    pub fn name(self) -> &'static str {
        match self {
            Tag::Ep => "EP",
            Tag::Ef => "EF",
            Tag::Ec => "EC",
            Tag::Etn => "ETN",
            Tag::Etm => "ETM",
            Tag::Ic => "IC",
            Tag::Jks => "JKS",
            Tag::Jkc => "JKC",
            Tag::Jkg => "JKG",
            Tag::Jko => "JKO",
            Tag::Jkb => "JKB",
            Tag::Jkv => "JKV",
            Tag::Jkq => "JKQ",
            Tag::Jx => "JX",
            Tag::Jc => "JC",
            Tag::Mag => "MAG",
            Tag::Maj => "MAJ",
            Tag::Mm => "MM",
            Tag::Nng => "NNG",
            Tag::Nnp => "NNP",
            Tag::Nnb => "NNB",
            Tag::Nnbc => "NNBC",
            Tag::Np => "NP",
            Tag::Nr => "NR",
            Tag::Sf => "SF",
            Tag::Sh => "SH",
            Tag::Sl => "SL",
            Tag::Sn => "SN",
            Tag::Sp => "SP",
            Tag::Ssc => "SSC",
            Tag::Sso => "SSO",
            Tag::Sc => "SC",
            Tag::Sy => "SY",
            Tag::Se => "SE",
            Tag::Va => "VA",
            Tag::Vcn => "VCN",
            Tag::Vcp => "VCP",
            Tag::Vv => "VV",
            Tag::Vx => "VX",
            Tag::Xpn => "XPN",
            Tag::Xr => "XR",
            Tag::Xsa => "XSA",
            Tag::Xsn => "XSN",
            Tag::Xsv => "XSV",
            Tag::Unknown => "UNKNOWN",
            Tag::Una => "UNA",
            Tag::Na => "NA",
            Tag::Vsv => "VSV",
        }
    }

    /// `code()`.
    pub fn code(self) -> i32 {
        match self {
            Tag::Ep => 100,
            Tag::Ef => 101,
            Tag::Ec => 102,
            Tag::Etn => 103,
            Tag::Etm => 104,
            Tag::Ic => 110,
            Tag::Jks => 120,
            Tag::Jkc => 121,
            Tag::Jkg => 122,
            Tag::Jko => 123,
            Tag::Jkb => 124,
            Tag::Jkv => 125,
            Tag::Jkq => 126,
            Tag::Jx => 127,
            Tag::Jc => 128,
            Tag::Mag => 130,
            Tag::Maj => 131,
            Tag::Mm => 140,
            Tag::Nng => 150,
            Tag::Nnp => 151,
            Tag::Nnb => 152,
            Tag::Nnbc => 153,
            Tag::Np => 154,
            Tag::Nr => 155,
            Tag::Sf => 160,
            Tag::Sh => 161,
            Tag::Sl => 162,
            Tag::Sn => 163,
            Tag::Sp => 164,
            Tag::Ssc => 165,
            Tag::Sso => 166,
            Tag::Sc => 167,
            Tag::Sy => 168,
            Tag::Se => 169,
            Tag::Va => 170,
            Tag::Vcn => 171,
            Tag::Vcp => 172,
            Tag::Vv => 173,
            Tag::Vx => 174,
            Tag::Xpn => 181,
            Tag::Xr => 182,
            Tag::Xsa => 183,
            Tag::Xsn => 184,
            Tag::Xsv => 185,
            Tag::Unknown => 999,
            Tag::Una | Tag::Na | Tag::Vsv => UNKNOWN_TAG_CODE,
        }
    }

    /// `description()`.
    pub fn description(self) -> &'static str {
        match self {
            Tag::Ep => "Pre-final ending",
            Tag::Ef => "Sentence-closing ending",
            Tag::Ec => "Connective ending",
            Tag::Etn => "Nominal transformative ending",
            Tag::Etm => "Adnominal form transformative ending",
            Tag::Ic => "Interjection",
            Tag::Jks => "Subject case marker",
            Tag::Jkc => "Complement case marker",
            Tag::Jkg => "Adnominal case marker",
            Tag::Jko => "Object case marker",
            Tag::Jkb => "Adverbial case marker",
            Tag::Jkv => "Vocative case marker",
            Tag::Jkq => "Quotative case marker",
            Tag::Jx => "Auxiliary postpositional particle",
            Tag::Jc => "Conjunctive postpositional particle",
            Tag::Mag => "General Adverb",
            Tag::Maj => "Conjunctive adverb",
            Tag::Mm => "Modifier",
            Tag::Nng => "General Noun",
            Tag::Nnp => "Proper Noun",
            Tag::Nnb => "Dependent noun",
            Tag::Nnbc => "Dependent noun",
            Tag::Np => "Pronoun",
            Tag::Nr => "Numeral",
            Tag::Sf => "Terminal punctuation",
            Tag::Sh => "Chinese Characeter",
            Tag::Sl => "Foreign language",
            Tag::Sn => "Number",
            Tag::Sp => "Space",
            Tag::Ssc => "Closing brackets",
            Tag::Sso => "Opening brackets",
            Tag::Sc => "Separator",
            Tag::Sy => "Other symbol",
            Tag::Se => "Ellipsis",
            Tag::Va => "Adjective",
            Tag::Vcn => "Negative designator",
            Tag::Vcp => "Positive designator",
            Tag::Vv => "Verb",
            Tag::Vx => "Auxiliary Verb or Adjective",
            Tag::Xpn => "Prefix",
            Tag::Xr => "Root",
            Tag::Xsa => "Adjective Suffix",
            Tag::Xsn => "Noun Suffix",
            Tag::Xsv => "Verb Suffix",
            Tag::Unknown => "Unknown",
            Tag::Una => "Unknown",
            Tag::Na => "Unknown",
            Tag::Vsv => "Unknown",
        }
    }

    /// `resolveTag(byte)`: `Tag.values()[tag]`, `None` out of range (Java:
    /// `ArrayIndexOutOfBoundsException`).
    pub fn resolve(tag: u8) -> Option<Tag> {
        TAGS.get(usize::from(tag)).copied()
    }

    /// `resolveTag(String)`: `Tag.valueOf(name.toUpperCase(Locale.ENGLISH))`.
    pub fn resolve_name(name: &str) -> Result<Tag, AnalysisError> {
        let upper = lucene_analysis::lang::java_string_to_upper_case(
            &name.encode_utf16().collect::<Vec<_>>(),
        );
        let upper = String::from_utf16_lossy(&upper);
        TAGS.iter()
            .copied()
            .find(|t| t.name() == upper)
            .ok_or_else(|| {
                AnalysisError::IllegalArgument(format!(
                    "No enum constant org.apache.lucene.analysis.ko.POS.Tag.{upper}"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_and_types_like_java() {
        assert_eq!(Tag::resolve(18), Some(Tag::Nng));
        assert_eq!(Tag::resolve(48), None);
        assert_eq!(Tag::Nng.code(), 150);
        assert_eq!(Tag::Sh.description(), "Chinese Characeter");
        assert_eq!(Tag::resolve_name("nng").unwrap(), Tag::Nng);
        assert!(Tag::resolve_name("x").is_err());
        assert_eq!(Tag::Vsv.name(), "VSV");
        assert_eq!(Type::resolve(6), Type::Inflect);
        assert_eq!(Type::resolve_name("*").unwrap(), Type::Morpheme);
        assert_eq!(Type::resolve_name("compound").unwrap(), Type::Compound);
        assert_eq!(Type::resolve_name("Inflect").unwrap().name(), "INFLECT");
        assert_eq!(
            Type::resolve_name("preanalysis").unwrap(),
            Type::Preanalysis
        );
        assert!(Type::resolve_name("x").is_err());
        assert_eq!(Type::Morpheme.name(), "MORPHEME");
        assert_eq!(Type::Preanalysis.name(), "PREANALYSIS");
    }

    #[test]
    fn every_tag_code_and_description() {
        let tags: Vec<Tag> = (0..48).map(|i| Tag::resolve(i).unwrap()).collect();
        // Java declares the tags in code order; UNA, NA and VSV share -1.
        let known: Vec<i32> = tags
            .iter()
            .map(|t| t.code())
            .filter(|&c| c != UNKNOWN_TAG_CODE)
            .collect();
        assert_eq!(known.len(), 45);
        assert!(known.windows(2).all(|w| w[0] < w[1]), "{known:?}");
        assert_eq!((known[0], known[44]), (100, 999));
        for t in &tags {
            assert!(!t.description().is_empty());
            assert_eq!(Tag::resolve_name(t.name()).unwrap(), *t);
        }
        assert_eq!(Tag::Una.code(), UNKNOWN_TAG_CODE);
    }
}
