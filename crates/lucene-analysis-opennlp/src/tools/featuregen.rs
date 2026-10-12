//! `opennlp.tools.util.featuregen`: the feature generators a POS model's
//! `generator.featuregen` descriptor configures (`GeneratorFactory.create`
//! over its DOM), and `StringPattern`/`FeatureGeneratorUtil`'s token
//! classes.
//!
//! The generator factories ported are those of OpenNLP's default POS
//! descriptor (`pos-default-features.xml`): `Definition`, `Suffix`,
//! `Prefix`, `Window`, `Token`, `Sentence`, `TokenClass` and `PosTagger`
//! (each `opennlp.tools.util.featuregen.<Name>FeatureGeneratorFactory`),
//! and `cache="true"`'s `CachedFeatureGenerator` -- which caches a
//! position's features by index alone, so the previous tags in them are
//! those of the first sequence the beam search scored there, as in Java.
//! Any other generator class (the name finder's dictionary, cluster and
//! brown generators, a user's own) is refused as an
//! `UnsupportedOperationException` when the tagger is created.

use std::collections::{HashMap, VecDeque};

use lucene_analysis::java_character as jc;
use lucene_analysis::java_unicode_script::{unicode_script_name, unicode_script_of};
use lucene_analysis::AnalysisError;

use super::supplementary_scripts::RUNS;
use super::text::{self, JString};
use super::xml::{self, Element, Node};

/// An error building the generators: Java's class, so that the message
/// says which (`InvalidFormatException` escaping `GeneratorFactory.create`
/// becomes `POSTaggerFactory`'s `IllegalStateException`; one inside a
/// nested generator is first wrapped in a `RuntimeException`).
#[derive(Debug)]
enum BuildError {
    InvalidFormat(String),
    Runtime(String),
    Analysis(AnalysisError),
}

impl BuildError {
    /// What `GeneratorFactory.create` throws out of `buildGenerator`.
    fn wrapped(self) -> BuildError {
        match self {
            BuildError::InvalidFormat(m) => {
                BuildError::Runtime(format!("opennlp.tools.util.InvalidFormatException: {m}"))
            }
            other => other,
        }
    }

    fn into_analysis(self) -> AnalysisError {
        match self {
            BuildError::InvalidFormat(m) => AnalysisError::IllegalState(format!(
                "IllegalStateException: opennlp.tools.util.InvalidFormatException: {m}"
            )),
            BuildError::Runtime(m) => AnalysisError::IllegalState(format!("RuntimeException: {m}")),
            BuildError::Analysis(e) => e,
        }
    }
}

/// An `AdaptiveFeatureGenerator`.
#[derive(Debug, Clone)]
pub enum Generator {
    /// `OutcomePriorFeatureGenerator` (`DefinitionFeatureGeneratorFactory`).
    Definition,
    /// `SuffixFeatureGenerator`.
    Suffix(usize),
    /// `PrefixFeatureGenerator`.
    Prefix(usize),
    /// `WindowFeatureGenerator`.
    Window {
        /// The windowed generator.
        inner: Box<Generator>,
        /// `prevWindowSize`.
        prev: i32,
        /// `nextWindowSize`.
        next: i32,
    },
    /// `TokenFeatureGenerator`.
    Token {
        /// Whether the token is lower-cased.
        lowercase: bool,
    },
    /// `SentenceFeatureGenerator`.
    Sentence {
        /// `S=begin` on the first token.
        begin: bool,
        /// `S=end` on the last.
        end: bool,
    },
    /// `TokenClassFeatureGenerator`.
    TokenClass {
        /// Also the word with its class.
        word_and_class: bool,
    },
    /// `PosTaggerFeatureGenerator`.
    PosTagger,
    /// `AggregatedFeatureGenerator`.
    Aggregated(Vec<Generator>),
    /// `CachedFeatureGenerator`.
    Cached(Box<Generator>, FeatureCache),
}

/// `CachedFeatureGenerator`'s cache: a position's features for the
/// current sentence, at most 100 (`Cache`, a `LinkedHashMap` dropping its
/// eldest entry).
#[derive(Debug, Clone, Default)]
pub struct FeatureCache {
    sentence: Option<u64>,
    map: HashMap<usize, Vec<JString>>,
    order: VecDeque<usize>,
}

const FEATURE_CACHE_CAPACITY: usize = 100;

/// A typed argument of a generator element.
#[derive(Debug, Clone)]
enum Arg {
    Int(i32),
    Bool(bool),
    Other,
    Generator(Generator),
}

fn text_of(el: &Element) -> Result<JString, BuildError> {
    match el.children.first() {
        Some(Node::Text(t)) => Ok(t.clone()),
        Some(Node::Element(_)) => Err(BuildError::Runtime(
            "ClassCastException: an argument element's first child is not text".into(),
        )),
        None => Err(BuildError::Runtime(
            "NullPointerException: an argument element has no text".into(),
        )),
    }
}

fn number_format(s: &[u16]) -> BuildError {
    BuildError::Runtime(format!(
        "NumberFormatException: For input string: \"{}\"",
        text::to_string(s)
    ))
}

/// `AbstractXmlFeatureGeneratorFactory.init`: the element's child
/// generators (`generator#<n>`, more than one aggregated as
/// `generator#0`) and typed arguments.
fn read_args(el: &Element) -> Result<HashMap<String, Arg>, BuildError> {
    let mut args = HashMap::new();
    let mut generators = Vec::new();
    for child in el.elements() {
        if child.name == "generator" {
            let g = build(child).map_err(BuildError::wrapped)?;
            args.insert(
                format!("generator#{}", generators.len()),
                Arg::Generator(g.clone()),
            );
            generators.push(g);
            continue;
        }
        let name = child
            .attribute("name")
            .map(|n| text::to_string(n))
            .unwrap_or_default();
        let value =
            match child.name.as_str() {
                "int" => {
                    let t = text_of(child)?;
                    Arg::Int(text::parse_int(&t).ok_or_else(|| number_format(&t))?)
                }
                "long" => {
                    let t = text_of(child)?;
                    text::to_string(&t)
                        .parse::<i64>()
                        .map_err(|_| number_format(&t))?;
                    Arg::Other
                }
                "float" | "double" => {
                    let t = text_of(child)?;
                    let s = text::to_string(text::trim(&t));
                    let s = s.trim_end_matches(['f', 'F', 'd', 'D']);
                    s.parse::<f64>().map_err(|_| number_format(&t))?;
                    Arg::Other
                }
                "str" => {
                    text_of(child)?;
                    Arg::Other
                }
                "bool" => Arg::Bool(text::parse_boolean(&text::to_string(&text_of(child)?))),
                _ => return Err(BuildError::InvalidFormat(
                    "child element must be one of generator, int, long, float, double, str or bool"
                        .into(),
                )),
            };
        args.insert(name, value);
    }
    if generators.len() > 1 {
        args.insert(
            "generator#0".into(),
            Arg::Generator(Generator::Aggregated(generators)),
        );
    }
    Ok(args)
}

fn get_int(
    args: &HashMap<String, Arg>,
    name: &str,
    default: Option<i32>,
) -> Result<i32, BuildError> {
    match (args.get(name), default) {
        (None, Some(d)) => Ok(d),
        (None, None) => Err(BuildError::InvalidFormat(format!(
            "parameter {name} must be set!"
        ))),
        (Some(Arg::Int(i)), _) => Ok(*i),
        (Some(_), _) => Err(BuildError::InvalidFormat(format!(
            "parameter {name} must be integer!"
        ))),
    }
}

fn get_bool(args: &HashMap<String, Arg>, name: &str, default: bool) -> Result<bool, BuildError> {
    match args.get(name) {
        None => Ok(default),
        Some(Arg::Bool(b)) => Ok(*b),
        Some(_) => Err(BuildError::InvalidFormat(format!(
            "parameter {name} must be boolean!"
        ))),
    }
}

/// `GeneratorFactory.buildGenerator`: a `<generator class="...">`.
fn build(el: &Element) -> Result<Generator, BuildError> {
    let class = el
        .attribute("class")
        .map(|c| text::to_string(c))
        .unwrap_or_default();
    if class.trim().is_empty() {
        return Err(BuildError::InvalidFormat(
            "generator must have class attribute".into(),
        ));
    }
    const PACKAGE: &str = "opennlp.tools.util.featuregen.";
    let simple = class.strip_prefix(PACKAGE).unwrap_or("");
    let known = [
        "DefinitionFeatureGeneratorFactory",
        "SuffixFeatureGeneratorFactory",
        "PrefixFeatureGeneratorFactory",
        "WindowFeatureGeneratorFactory",
        "TokenFeatureGeneratorFactory",
        "SentenceFeatureGeneratorFactory",
        "TokenClassFeatureGeneratorFactory",
        "PosTaggerFeatureGeneratorFactory",
    ];
    if !known.contains(&simple) {
        return Err(BuildError::Analysis(AnalysisError::UnsupportedOperation(
            format!("the feature generator {class} is not supported by this port"),
        )));
    }
    let args = read_args(el)?;
    let length = |name: &str| -> Result<usize, BuildError> {
        // `min(length, lex.length())` of a negative length is no feature
        Ok(usize::try_from(get_int(&args, name, Some(4))?).unwrap_or(0))
    };
    Ok(match simple {
        "DefinitionFeatureGeneratorFactory" => Generator::Definition,
        "SuffixFeatureGeneratorFactory" => Generator::Suffix(length("length")?),
        "PrefixFeatureGeneratorFactory" => Generator::Prefix(length("length")?),
        "WindowFeatureGeneratorFactory" => {
            let inner = match args.get("generator#0") {
                Some(Arg::Generator(g)) => g.clone(),
                _ => {
                    return Err(BuildError::InvalidFormat(
                        "window feature generator must contain an aggregator element".into(),
                    ))
                }
            };
            Generator::Window {
                inner: Box::new(inner),
                prev: get_int(&args, "prevLength", None)?,
                next: get_int(&args, "nextLength", None)?,
            }
        }
        "TokenFeatureGeneratorFactory" => Generator::Token {
            lowercase: get_bool(&args, "lowercase", true)?,
        },
        "SentenceFeatureGeneratorFactory" => Generator::Sentence {
            begin: get_bool(&args, "begin", true)?,
            end: get_bool(&args, "end", true)?,
        },
        "TokenClassFeatureGeneratorFactory" => Generator::TokenClass {
            word_and_class: get_bool(&args, "wordAndClass", true)?,
        },
        _ => Generator::PosTagger,
    })
}

/// `GeneratorFactory.create(descriptor)` as `POSTaggerFactory`'s
/// `createFeatureGenerators` calls it.
pub fn create(descriptor: &[u8]) -> Result<Generator, AnalysisError> {
    let root = xml::parse(descriptor).map_err(|e| {
        AnalysisError::IllegalState(format!(
            "IllegalStateException: opennlp.tools.util.InvalidFormatException: Descriptor is not valid XML! {}",
            e.message
        ))
    })?;
    if root.name != "featureGenerators" {
        return Err(AnalysisError::IllegalArgument(
            "[OPENNLP-1174] - Classic configuration format is no longer supported!".into(),
        ));
    }
    let build_all = || -> Result<Generator, BuildError> {
        let mut generators = Vec::new();
        for child in root.elements() {
            if child.name != "generator" {
                return Err(BuildError::InvalidFormat(
                    "Unexpected element: featureGenerators".into(),
                ));
            }
            generators.push(build(child).map_err(BuildError::wrapped)?);
        }
        let generator = match generators.len() {
            0 => {
                return Err(BuildError::InvalidFormat(
                    "featureGenerators must have one or more generators".into(),
                ))
            }
            1 => generators.pop().unwrap_or(Generator::Definition),
            _ => Generator::Aggregated(generators),
        };
        let cache = root.attribute("cache").map(|c| text::to_string(c));
        Ok(if cache.as_deref().is_some_and(text::parse_boolean) {
            Generator::Cached(Box::new(generator), FeatureCache::default())
        } else {
            generator
        })
    };
    build_all().map_err(BuildError::into_analysis)
}

fn cat(prefix: &str, rest: &[u16]) -> JString {
    text::concat(&[&text::units(prefix), rest])
}

impl Generator {
    /// `createFeatures(features, tokens, index, previousOutcomes)`;
    /// `sentence` identifies `tokens` (Java compares the array by
    /// identity). Fails where Java indexes past the previous outcomes.
    // ARITH: token indices and lengths of in-memory strings; window offsets are checked against the sentence.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn create_features(
        &mut self,
        features: &mut Vec<JString>,
        tokens: &[JString],
        index: usize,
        prior: &[JString],
        sentence: u64,
    ) -> Result<(), AnalysisError> {
        match self {
            Generator::Definition => features.push(text::units("def")),
            Generator::Suffix(n) => {
                let lex = &tokens[index];
                for li in 0..(*n).min(lex.len()) {
                    features.push(cat("suf=", &lex[lex.len() - li - 1..]));
                }
            }
            Generator::Prefix(n) => {
                let lex = &tokens[index];
                for li in 0..(*n).min(lex.len()) {
                    features.push(cat("pre=", &lex[..li + 1]));
                }
            }
            Generator::Window { inner, prev, next } => {
                inner.create_features(features, tokens, index, prior, sentence)?;
                let mut window = Vec::new();
                // Java tries every offset of the window; past the
                // sentence's ends none adds a feature, so the loops stop
                // there (a window of `Integer.MAX_VALUE` costs nothing)
                for i in 1..=((*prev).max(0) as usize).min(index) {
                    window.clear();
                    inner.create_features(&mut window, tokens, index - i, prior, sentence)?;
                    let p = format!("p{i}");
                    features.extend(window.iter().map(|f| cat(&p, f)));
                }
                let ahead = tokens.len().saturating_sub(index + 1);
                for i in 1..=((*next).max(0) as usize).min(ahead) {
                    window.clear();
                    inner.create_features(&mut window, tokens, index + i, prior, sentence)?;
                    let n = format!("n{i}");
                    features.extend(window.iter().map(|f| cat(&n, f)));
                }
            }
            Generator::Token { lowercase } => {
                if *lowercase {
                    features.push(cat("w=", &text::to_lower_case(&tokens[index])));
                } else {
                    features.push(cat("w=", &tokens[index]));
                }
            }
            Generator::Sentence { begin, end } => {
                if *begin && index == 0 {
                    features.push(text::units("S=begin"));
                }
                if *end && tokens.len() == index + 1 {
                    features.push(text::units("S=end"));
                }
            }
            Generator::TokenClass { word_and_class } => {
                let class = token_feature(&tokens[index]);
                features.push(cat("wc=", &text::units(class)));
                if *word_and_class {
                    let mut f = cat("w&c=", &text::to_lower_case(&tokens[index]));
                    f.push(u16::from(b','));
                    f.extend(class.encode_utf16());
                    features.push(f);
                }
            }
            Generator::PosTagger => {
                if index >= 1 {
                    let tag = |i: usize| {
                        prior.get(i).ok_or_else(|| {
                            AnalysisError::IllegalState(format!(
                                "ArrayIndexOutOfBoundsException: Index {i} out of bounds for length {}",
                                prior.len()
                            ))
                        })
                    };
                    let tagprev = tag(index - 1)?;
                    features.push(cat("t=", tagprev));
                    if index >= 2 {
                        let mut f = cat("t2=", tag(index - 2)?);
                        f.push(u16::from(b','));
                        f.extend_from_slice(tagprev);
                        features.push(f);
                    }
                }
            }
            Generator::Aggregated(gens) => {
                for g in gens {
                    g.create_features(features, tokens, index, prior, sentence)?;
                }
            }
            Generator::Cached(inner, cache) => {
                if cache.sentence == Some(sentence) {
                    if let Some(cached) = cache.map.get(&index) {
                        features.extend_from_slice(cached);
                        return Ok(());
                    }
                } else {
                    cache.map.clear();
                    cache.order.clear();
                    cache.sentence = Some(sentence);
                }
                let mut computed = Vec::new();
                inner.create_features(&mut computed, tokens, index, prior, sentence)?;
                features.extend_from_slice(&computed);
                cache.map.insert(index, computed);
                cache.order.push_back(index);
                if cache.map.len() > FEATURE_CACHE_CAPACITY {
                    if let Some(eldest) = cache.order.pop_front() {
                        cache.map.remove(&eldest);
                    }
                }
            }
        }
        Ok(())
    }
}

/// The scripts `StringPattern.recognize` distinguishes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Script {
    Common,
    Latin,
    Han,
    Hiragana,
    Katakana,
    Other,
}

/// `Character.UnicodeScript.of(cp)`, as far as [`Script`] tells.
fn script_of(cp: u32) -> Script {
    let by_name = |name: &str| match name {
        "COMMON" => Script::Common,
        "LATIN" => Script::Latin,
        "HAN" => Script::Han,
        "HIRAGANA" => Script::Hiragana,
        "KATAKANA" => Script::Katakana,
        _ => Script::Other,
    };
    match u16::try_from(cp) {
        Ok(unit) => by_name(unicode_script_name(unicode_script_of(unit))),
        Err(_) => {
            let i = RUNS.partition_point(|&(first, _, _)| first <= cp);
            match i.checked_sub(1).map(|i| RUNS[i]) {
                Some((_, last, s)) if cp <= last => match s {
                    0 => Script::Common,
                    1 => Script::Latin,
                    2 => Script::Han,
                    3 => Script::Hiragana,
                    _ => Script::Katakana,
                },
                _ => Script::Other,
            }
        }
    }
}

const INITIAL_CAPITAL_LETTER: u32 = 0x1;
const ALL_CAPITAL_LETTER: u32 = 0x1 << 1;
const ALL_LOWERCASE_LETTER: u32 = 0x1 << 2;
const ALL_LETTERS: u32 = 0x1 << 3;
const ALL_DIGIT: u32 = 0x1 << 4;
const ALL_HIRAGANA: u32 = 0x1 << 5;
const ALL_KATAKANA: u32 = 0x1 << 6;
const CONTAINS_PERIOD: u32 = 0x1 << 7;
const CONTAINS_COMMA: u32 = 0x1 << 8;
const CONTAINS_SLASH: u32 = 0x1 << 9;
const CONTAINS_DIGIT: u32 = 0x1 << 10;
const CONTAINS_HYPHEN: u32 = 0x1 << 11;
const CONTAINS_LETTERS: u32 = 0x1 << 12;
const CONTAINS_UPPERCASE: u32 = 0x1 << 13;

/// `StringPattern.recognize(token)`: the pattern bits and the digit count.
// ARITH: the digit count is at most the token's length.
#[allow(clippy::arithmetic_side_effects)]
fn recognize(token: &[u16]) -> (u32, usize) {
    let mut pattern = ALL_CAPITAL_LETTER
        | ALL_LOWERCASE_LETTER
        | ALL_DIGIT
        | ALL_LETTERS
        | ALL_HIRAGANA
        | ALL_KATAKANA;
    let mut digits = 0;
    for (i, &ch) in token.iter().enumerate() {
        let letter_type = jc::get_type(u32::from(ch));
        let is_letter = matches!(
            letter_type,
            jc::UPPERCASE_LETTER
                | jc::LOWERCASE_LETTER
                | jc::TITLECASE_LETTER
                | jc::MODIFIER_LETTER
                | jc::OTHER_LETTER
        );
        if is_letter {
            pattern |= CONTAINS_LETTERS;
            pattern &= !ALL_DIGIT;
            if letter_type == jc::UPPERCASE_LETTER {
                if i == 0 {
                    pattern |= INITIAL_CAPITAL_LETTER;
                }
                pattern |= CONTAINS_UPPERCASE;
                pattern &= !ALL_LOWERCASE_LETTER;
            } else {
                pattern &= !ALL_CAPITAL_LETTER;
            }
        } else {
            pattern &= !(ALL_LETTERS | ALL_CAPITAL_LETTER | ALL_LOWERCASE_LETTER);
            if letter_type == jc::DECIMAL_DIGIT_NUMBER {
                pattern |= CONTAINS_DIGIT;
                pattern &= !(ALL_HIRAGANA | ALL_KATAKANA);
                digits += 1;
            } else {
                pattern &= !ALL_DIGIT;
            }
            match ch {
                0x2C => pattern |= CONTAINS_COMMA,
                0x2E => pattern |= CONTAINS_PERIOD,
                0x2F => pattern |= CONTAINS_SLASH,
                0x2D => pattern |= CONTAINS_HYPHEN,
                _ => {}
            }
        }
        // `token.codePointAt(i)`: a pair read whole at its high surrogate
        let cp = text::code_points(&token[i..])
            .next()
            .unwrap_or(u32::from(ch));
        match script_of(cp) {
            Script::Latin => pattern &= !(ALL_HIRAGANA | ALL_KATAKANA),
            Script::Han => pattern &= !(ALL_HIRAGANA | ALL_KATAKANA | ALL_LOWERCASE_LETTER),
            Script::Hiragana => pattern &= !(ALL_KATAKANA | ALL_LOWERCASE_LETTER),
            Script::Katakana => pattern &= !(ALL_HIRAGANA | ALL_LOWERCASE_LETTER),
            Script::Common => {
                if ch != 0x30FB && ch != 0x30FC && ch != 0x301C {
                    pattern &= !(ALL_HIRAGANA | ALL_KATAKANA);
                }
            }
            Script::Other => {}
        }
    }
    (pattern, digits)
}

/// `^[A-ZÄÖÜ]\.$` found in `token`: the capital and the period, then the
/// end or a final line terminator (Java's `$`).
fn cap_period(token: &[u16]) -> bool {
    let body = match token {
        [rest @ .., 0x0D, 0x0A] => rest,
        [rest @ .., 0x0A | 0x0D | 0x85 | 0x2028 | 0x2029] => rest,
        _ => token,
    };
    let matches = |t: &[u16]| matches!(t, [c, 0x2E] if (u16::from(b'A')..=u16::from(b'Z')).contains(c) || matches!(*c, 0xC4 | 0xD6 | 0xDC));
    matches(token) || matches(body)
}

/// `FeatureGeneratorUtil.tokenFeature(token)`.
pub fn token_feature(token: &[u16]) -> &'static str {
    let (p, digits) = recognize(token);
    if p & ALL_HIRAGANA != 0 {
        "jah"
    } else if p & ALL_KATAKANA != 0 {
        "jak"
    } else if p & ALL_LOWERCASE_LETTER != 0 {
        "lc"
    } else if digits == 2 {
        "2d"
    } else if digits == 4 {
        "4d"
    } else if p & CONTAINS_DIGIT != 0 {
        if p & CONTAINS_LETTERS != 0 {
            "an"
        } else if p & CONTAINS_HYPHEN != 0 {
            "dd"
        } else if p & CONTAINS_SLASH != 0 {
            "ds"
        } else if p & CONTAINS_COMMA != 0 {
            "dc"
        } else if p & CONTAINS_PERIOD != 0 {
            "dp"
        } else {
            "num"
        }
    } else if p & ALL_CAPITAL_LETTER != 0 {
        if token.len() == 1 {
            "sc"
        } else {
            "ac"
        }
    } else if cap_period(token) {
        "cp"
    } else if p & INITIAL_CAPITAL_LETTER != 0 {
        "ic"
    } else {
        "other"
    }
}
