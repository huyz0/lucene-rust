//! `opennlp.tools.util.featuregen`: the feature generators a model's
//! `generator.featuregen` descriptor configures (`GeneratorFactory.create`
//! over its DOM, resources looked up among the model's artifacts), and
//! `StringPattern`/`FeatureGeneratorUtil`'s token classes.
//!
//! Every generator factory OpenNLP 2.5.3 ships is ported (each
//! `opennlp.tools.util.featuregen.<Name>FeatureGeneratorFactory`): the POS
//! tagger's `Definition`, `Suffix`, `Prefix`, `Window`, `Token`,
//! `Sentence`, `TokenClass` and `PosTagger`; the name finder's
//! `PreviousMap`, `BigramName`, `TrigramName`, `DocumentBegin`,
//! `CharacterNgram`, `TokenPattern` (over `SimpleTokenizer`),
//! `Dictionary` (an `InSpanGenerator` over a `DictionaryNameFinder`),
//! `WordCluster`, `BrownClusterToken`/`TokenClass`/`Bigram` and
//! `POSTaggerName` (a nested POS model's own tags); and `Aggregated` and
//! `Cached`. `cache="true"` (or `Cached`) caches a position's features by
//! index alone, so the previous tags in them are those of the first
//! sequence the beam search scored there, as in Java.
//!
//! A generator's identity checks on the token array (the cache, the
//! dictionary's names, `DocumentBegin`'s first sentence) are a sentence
//! number here: every call of a tool is a new sentence, as every array a
//! Lucene op passes is a new one.
//!
//! Refused (`UnsupportedOperationException`): any other generator class (a
//! user's own, which Java loads by reflection). Differs:
//! `Character.isAlphabetic` (`TokenPattern`) takes the JDK's general
//! categories but Rust's `Other_Alphabetic` property for marks and symbols.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use lucene_analysis::java_character as jc;
use lucene_analysis::java_unicode_script::{unicode_script_name, unicode_script_of};
use lucene_analysis::AnalysisError;

use super::dictionary::{ClusterDictionary, Dictionary};
use super::model::{Artifact, BaseModel};
use super::postag::PosTagger;
use super::supplementary_scripts::RUNS;
use super::text::{self, JString};
use super::xml::{self, Element, Node};

/// Who builds the generators: the tool factory decides how an
/// `InvalidFormatException` out of `GeneratorFactory.create` surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    /// `POSTaggerFactory`: an `IllegalStateException` around it.
    PosTagger,
    /// `TokenNameFinderFactory`: a `FeatureGeneratorCreationError` around
    /// it.
    NameFinder,
}

/// `FeatureGeneratorResourceProvider`: a resource by its key (the model's
/// artifacts).
pub trait Resources {
    /// `getResource(key)`.
    fn resource(&self, key: &str) -> Option<&Artifact>;
}

impl Resources for BaseModel {
    fn resource(&self, key: &str) -> Option<&Artifact> {
        self.artifact(key)
    }
}

/// A provider without resources (`key -> null`).
pub struct NoResources;

impl Resources for NoResources {
    fn resource(&self, _key: &str) -> Option<&Artifact> {
        None
    }
}

/// An error building the generators: Java's class, so that the message
/// says which (`InvalidFormatException` escaping `GeneratorFactory.create`
/// is wrapped by the tool factory; one inside a factory is first wrapped
/// in a `RuntimeException` by `buildGenerator`; anything else escapes as
/// it is).
#[derive(Debug)]
enum BuildError {
    InvalidFormat(String),
    Runtime(String),
    Analysis(AnalysisError),
}

impl BuildError {
    /// What `buildGenerator`'s `catch` makes of an error inside a factory.
    fn wrapped(self) -> BuildError {
        match self {
            BuildError::InvalidFormat(m) => {
                BuildError::Runtime(format!("opennlp.tools.util.InvalidFormatException: {m}"))
            }
            other => other,
        }
    }

    fn into_analysis(self, owner: Owner) -> AnalysisError {
        match self {
            BuildError::InvalidFormat(m) => AnalysisError::IllegalState(match owner {
                Owner::PosTagger => format!(
                    "IllegalStateException: opennlp.tools.util.InvalidFormatException: {m}"
                ),
                Owner::NameFinder => format!(
                    "TokenNameFinderModel.FeatureGeneratorCreationError: opennlp.tools.util.InvalidFormatException: {m}"
                ),
            }),
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
    /// `PreviousMapFeatureGenerator`: each token's outcome in the
    /// sentences tagged since the adaptive data was last cleared.
    PreviousMap(HashMap<JString, JString>),
    /// `BigramNameFeatureGenerator`.
    BigramName,
    /// `TrigramNameFeatureGenerator`.
    TrigramName,
    /// `DocumentBeginFeatureGenerator`: the first sentence it saw since the
    /// adaptive data was last cleared.
    DocumentBegin(Option<u64>),
    /// `CharacterNgramFeatureGenerator`.
    CharacterNgram {
        /// `minLength`.
        min: i32,
        /// `maxLength`.
        max: i32,
    },
    /// `TokenPatternFeatureGenerator` over `SimpleTokenizer`.
    TokenPattern,
    /// `DictionaryFeatureGenerator`: an `InSpanGenerator` (prefix `""`)
    /// over a `DictionaryNameFinder`, the current sentence's names cached.
    Dictionary {
        /// The dictionary.
        dictionary: Arc<Dictionary>,
        /// The sentence `names` are of.
        sentence: Option<u64>,
        /// Its names, `[start, end)` token spans.
        names: Vec<(usize, usize)>,
    },
    /// `WordClusterFeatureGenerator`.
    WordCluster {
        /// The clusters.
        clusters: Arc<ClusterDictionary>,
        /// The resource's key, the feature's prefix.
        name: JString,
        /// Whether the token is lower-cased before its lookup.
        lowercase: bool,
    },
    /// `BrownTokenFeatureGenerator`.
    BrownToken(Arc<ClusterDictionary>),
    /// `BrownTokenClassFeatureGenerator`.
    BrownTokenClass(Arc<ClusterDictionary>),
    /// `BrownBigramFeatureGenerator`.
    BrownBigram(Arc<ClusterDictionary>),
    /// `POSTaggerNameFeatureGenerator`: a nested POS model's tags, the
    /// last sentence's kept while the next has the same tokens.
    PosTaggerName {
        /// The tagger.
        tagger: Box<PosTagger>,
        /// The last sentence's tokens and tags.
        cached: Option<(Vec<JString>, Vec<JString>)>,
    },
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
    Str(JString),
    Other,
    Generator(Generator),
}

/// `AbstractXmlFeatureGeneratorFactory.args`: a `LinkedHashMap`, so a
/// repeated name keeps its first place.
#[derive(Debug, Default)]
struct Args(Vec<(String, Arg)>);

impl Args {
    fn put(&mut self, name: String, value: Arg) {
        match self.0.iter_mut().find(|(n, _)| *n == name) {
            Some((_, v)) => *v = value,
            None => self.0.push((name, value)),
        }
    }

    fn get(&self, name: &str) -> Option<&Arg> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
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
fn read_args(el: &Element, resources: &dyn Resources) -> Result<Args, BuildError> {
    let mut args = Args::default();
    let mut generators = Vec::new();
    for child in el.elements() {
        if child.name == "generator" {
            let g = build(child, resources)?;
            args.put(
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
                "str" => Arg::Str(text_of(child)?),
                "bool" => Arg::Bool(text::parse_boolean(&text::to_string(&text_of(child)?))),
                _ => return Err(BuildError::InvalidFormat(
                    "child element must be one of generator, int, long, float, double, str or bool"
                        .into(),
                )),
            };
        args.put(name, value);
    }
    if generators.len() > 1 {
        args.put(
            "generator#0".into(),
            Arg::Generator(Generator::Aggregated(generators)),
        );
    }
    Ok(args)
}

fn get_int(args: &Args, name: &str, default: Option<i32>) -> Result<i32, BuildError> {
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

fn get_bool(args: &Args, name: &str, default: Option<bool>) -> Result<bool, BuildError> {
    match (args.get(name), default) {
        (None, Some(d)) => Ok(d),
        (None, None) => Err(BuildError::InvalidFormat(format!(
            "parameter {name} must be set!"
        ))),
        (Some(Arg::Bool(b)), _) => Ok(*b),
        (Some(_), _) => Err(BuildError::InvalidFormat(format!(
            "parameter {name} must be boolean!"
        ))),
    }
}

/// `getStr(name)` (without a default; its type error says "double", as
/// Java's does).
fn get_str(args: &Args, name: &str) -> Result<JString, BuildError> {
    match args.get(name) {
        None => Err(BuildError::InvalidFormat(format!(
            "parameter {name} must be set!"
        ))),
        Some(Arg::Str(s)) => Ok(s.clone()),
        Some(_) => Err(BuildError::InvalidFormat(format!(
            "parameter {name} must be double!"
        ))),
    }
}

/// The resource a factory's `str` argument `name` names: its key and the
/// artifact, if any.
fn lookup<'r>(
    args: &Args,
    name: &str,
    resources: &'r dyn Resources,
) -> Result<(JString, Option<&'r Artifact>), BuildError> {
    let key = get_str(args, name)?;
    let artifact = resources.resource(&text::to_string(&key));
    Ok((key, artifact))
}

/// The factory class names this port builds, without their package.
const FACTORIES: [&str; 22] = [
    "AggregatedFeatureGeneratorFactory",
    "BigramNameFeatureGeneratorFactory",
    "BrownClusterBigramFeatureGeneratorFactory",
    "BrownClusterTokenClassFeatureGeneratorFactory",
    "BrownClusterTokenFeatureGeneratorFactory",
    "CachedFeatureGeneratorFactory",
    "CharacterNgramFeatureGeneratorFactory",
    "DefinitionFeatureGeneratorFactory",
    "DictionaryFeatureGeneratorFactory",
    "DocumentBeginFeatureGeneratorFactory",
    "POSTaggerNameFeatureGeneratorFactory",
    "PosTaggerFeatureGeneratorFactory",
    "PrefixFeatureGeneratorFactory",
    "PreviousMapFeatureGeneratorFactory",
    "SentenceFeatureGeneratorFactory",
    "SuffixFeatureGeneratorFactory",
    "TokenClassFeatureGeneratorFactory",
    "TokenFeatureGeneratorFactory",
    "TokenPatternFeatureGeneratorFactory",
    "TrigramNameFeatureGeneratorFactory",
    "WindowFeatureGeneratorFactory",
    "WordClusterFeatureGeneratorFactory",
];

/// `GeneratorFactory.buildGenerator`: a `<generator class="...">`. A
/// missing class is thrown as it is; what the factory's `init` and
/// `create` throw is wrapped.
fn build(el: &Element, resources: &dyn Resources) -> Result<Generator, BuildError> {
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
    if !FACTORIES.contains(&simple) {
        return Err(BuildError::Analysis(AnalysisError::UnsupportedOperation(
            format!("the feature generator {class} is not supported by this port"),
        )));
    }
    build_factory(simple, el, resources).map_err(BuildError::wrapped)
}

/// The factory's `init(element)` and `create()`.
fn build_factory(
    simple: &str,
    el: &Element,
    resources: &dyn Resources,
) -> Result<Generator, BuildError> {
    let args = read_args(el, resources)?;
    let length = |name: &str| -> Result<usize, BuildError> {
        // `min(length, lex.length())` of a negative length is no feature
        Ok(usize::try_from(get_int(&args, name, Some(4))?).unwrap_or(0))
    };
    let brown = |args: &Args| -> Result<Arc<ClusterDictionary>, BuildError> {
        match lookup(args, "dict", resources)? {
            (_, Some(Artifact::BrownCluster(c))) => Ok(Arc::clone(c)),
            (key, _) => Err(BuildError::InvalidFormat(format!(
                "Not a BrownLexicon resource for key: {}",
                text::to_string(&key)
            ))),
        }
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
            lowercase: get_bool(&args, "lowercase", Some(true))?,
        },
        "SentenceFeatureGeneratorFactory" => Generator::Sentence {
            begin: get_bool(&args, "begin", Some(true))?,
            end: get_bool(&args, "end", Some(true))?,
        },
        "TokenClassFeatureGeneratorFactory" => Generator::TokenClass {
            word_and_class: get_bool(&args, "wordAndClass", Some(true))?,
        },
        "PosTaggerFeatureGeneratorFactory" => Generator::PosTagger,
        "PreviousMapFeatureGeneratorFactory" => Generator::PreviousMap(HashMap::new()),
        "BigramNameFeatureGeneratorFactory" => Generator::BigramName,
        "TrigramNameFeatureGeneratorFactory" => Generator::TrigramName,
        "DocumentBeginFeatureGeneratorFactory" => Generator::DocumentBegin(None),
        "TokenPatternFeatureGeneratorFactory" => Generator::TokenPattern,
        "CharacterNgramFeatureGeneratorFactory" => Generator::CharacterNgram {
            min: get_int(&args, "min", None)?,
            max: get_int(&args, "max", None)?,
        },
        "AggregatedFeatureGeneratorFactory" => {
            // every `generator#` argument in the map's order: with more
            // than one child, `generator#0` is already their aggregate
            let mut generators = Vec::new();
            for (name, value) in &args.0 {
                if !name.starts_with("generator#") {
                    continue;
                }
                match value {
                    Arg::Generator(g) => generators.push(g.clone()),
                    _ => {
                        return Err(BuildError::Runtime(format!(
                            "ClassCastException: the argument {name} is not a generator"
                        )))
                    }
                }
            }
            Generator::Aggregated(generators)
        }
        "CachedFeatureGeneratorFactory" => match args.get("generator#0") {
            Some(Arg::Generator(g)) => {
                Generator::Cached(Box::new(g.clone()), FeatureCache::default())
            }
            _ => {
                return Err(BuildError::InvalidFormat(
                    "Could not find containing generator element!".into(),
                ))
            }
        },
        "DictionaryFeatureGeneratorFactory" => match lookup(&args, "dict", resources)? {
            (_, Some(Artifact::Dictionary(d))) => Generator::Dictionary {
                dictionary: Arc::clone(d),
                sentence: None,
                names: Vec::new(),
            },
            (key, _) => {
                return Err(BuildError::InvalidFormat(format!(
                    "No dictionary resource for key: {}",
                    text::to_string(&key)
                )))
            }
        },
        "WordClusterFeatureGeneratorFactory" => {
            let key = get_str(&args, "dict")?;
            let lowercase = get_bool(&args, "lowerCase", None)?;
            match resources.resource(&text::to_string(&key)) {
                Some(Artifact::WordClusters(c)) => Generator::WordCluster {
                    clusters: Arc::clone(c),
                    name: key,
                    lowercase,
                },
                _ => {
                    return Err(BuildError::InvalidFormat(format!(
                        "Not a WordClusterDictionary resource for key: {}",
                        text::to_string(&key)
                    )))
                }
            }
        }
        "BrownClusterTokenFeatureGeneratorFactory" => Generator::BrownToken(brown(&args)?),
        "BrownClusterTokenClassFeatureGeneratorFactory" => {
            Generator::BrownTokenClass(brown(&args)?)
        }
        "BrownClusterBigramFeatureGeneratorFactory" => Generator::BrownBigram(brown(&args)?),
        _ => {
            // POSTaggerNameFeatureGeneratorFactory: `(POSModel)` cast of
            // the resource, then `new POSTaggerME(model, guessFormat(model))`
            // -- their exceptions escape unwrapped
            let model = match lookup(&args, "model", resources)? {
                (_, Some(Artifact::PosModel(m))) => Arc::clone(m),
                (_, None) => {
                    return Err(BuildError::Analysis(AnalysisError::IllegalState(
                        "NullPointerException: POSModel must not be NULL.".into(),
                    )))
                }
                (key, Some(a)) => {
                    return Err(BuildError::Analysis(AnalysisError::IllegalState(format!(
                        "ClassCastException: the resource {} is not a POSModel but {}",
                        text::to_string(&key),
                        a.describe()
                    ))))
                }
            };
            let tagger = PosTagger::new_untranslated(&model).map_err(BuildError::Analysis)?;
            Generator::PosTaggerName {
                tagger: Box::new(tagger),
                cached: None,
            }
        }
    })
}

/// `GeneratorFactory.create(descriptor, key -> null)` as
/// `POSTaggerFactory` wraps its errors.
pub fn create(descriptor: &[u8]) -> Result<Generator, AnalysisError> {
    create_with(descriptor, &NoResources, Owner::PosTagger)
}

/// `GeneratorFactory.create(descriptor, resources)` as `owner`'s
/// `createFeatureGenerators` calls it.
pub fn create_with(
    descriptor: &[u8],
    resources: &dyn Resources,
    owner: Owner,
) -> Result<Generator, AnalysisError> {
    let root = xml::parse(descriptor).map_err(|e| {
        BuildError::InvalidFormat(format!("Descriptor is not valid XML! {}", e.message))
            .into_analysis(owner)
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
            generators.push(build(child, resources)?);
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
        if matches!(generator, Generator::Cached(..)) {
            return Err(BuildError::InvalidFormat(
                "CachedFeatureGeneratorFactory cannot be specified manually.Use cache=\"true\" attribute in featureGenerators element instead."
                    .into(),
            ));
        }
        let cache = root.attribute("cache").map(|c| text::to_string(c));
        Ok(if cache.as_deref().is_some_and(text::parse_boolean) {
            Generator::Cached(Box::new(generator), FeatureCache::default())
        } else {
            generator
        })
    };
    build_all().map_err(|e| e.into_analysis(owner))
}

fn cat(prefix: &str, rest: &[u16]) -> JString {
    text::keyed(prefix, rest)
}

/// `a + "," + b`.
fn comma(prefix: &str, parts: &[&[u16]]) -> JString {
    let mut f = text::units(prefix);
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            f.push(u16::from(b','));
        }
        f.extend_from_slice(p);
    }
    f
}

/// `String.valueOf(map.get(key))`: `"null"` for an absent key.
fn or_null(v: Option<&JString>) -> JString {
    v.cloned().unwrap_or_else(|| text::units("null"))
}

/// `BrownTokenClasses.getWordClasses`: the cluster's bit-string prefixes
/// of 4, 6, 10 and 20 units, each only while the previous was shorter
/// than the whole.
fn word_classes(token: &[u16], clusters: &ClusterDictionary) -> Vec<JString> {
    const PATH_LENGTHS: [usize; 4] = [4, 6, 10, 20];
    let Some(class) = clusters.lookup(token) else {
        return Vec::new();
    };
    let mut classes = vec![class[..class.len().min(PATH_LENGTHS[0])].to_vec()];
    for w in PATH_LENGTHS.windows(2) {
        if w[0] < class.len() {
            classes.push(class[..class.len().min(w[1])].to_vec());
        }
    }
    classes
}

/// `DictionaryNameFinder.find`: from each token, the longest entry
/// starting there (no longer than the longest entry), the search going on
/// after it.
// ARITH: token offsets below the sentence's length.
#[allow(clippy::arithmetic_side_effects)]
fn dictionary_names(dictionary: &Dictionary, tokens: &[JString]) -> Vec<(usize, usize)> {
    let mut names = Vec::new();
    let mut from = 0;
    while from < tokens.len() {
        let mut found = None;
        for to in from..tokens.len() {
            if to - from + 1 > dictionary.max_token_count() {
                break;
            }
            if dictionary.contains(&tokens[from..=to]) {
                found = Some((from, to + 1));
            }
        }
        if let Some((start, end)) = found {
            names.push((start, end));
            from += end - start - 1;
        }
        from += 1;
    }
    names
}

/// `Character.isAlphabetic(c)` of a UTF-16 unit: the letters and letter
/// numbers by the JDK's categories, `Other_Alphabetic` (marks and
/// symbols) by Rust's.
fn is_alphabetic(c: u16) -> bool {
    match jc::get_type(u32::from(c)) {
        jc::UPPERCASE_LETTER
        | jc::LOWERCASE_LETTER
        | jc::TITLECASE_LETTER
        | jc::MODIFIER_LETTER
        | jc::OTHER_LETTER
        | jc::LETTER_NUMBER => true,
        jc::NON_SPACING_MARK | jc::COMBINING_SPACING_MARK | jc::OTHER_SYMBOL => {
            char::from_u32(u32::from(c)).is_some_and(char::is_alphabetic)
        }
        _ => false,
    }
}

/// `SimpleTokenizer.INSTANCE.tokenize(s)`: runs of white space split off,
/// letters, digits and other characters apart, each different other
/// character on its own.
// ARITH: offsets within the string.
#[allow(clippy::arithmetic_side_effects)]
pub fn simple_tokenize(s: &[u16]) -> Vec<JString> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Whitespace,
        Alphabetic,
        Numeric,
        Other,
    }
    let mut tokens = Vec::new();
    let mut kind = Kind::Whitespace;
    let mut state = kind;
    let mut start = 0;
    let mut pc = 0u16;
    for (ci, &c) in s.iter().enumerate() {
        kind = if text::is_whitespace(c) {
            Kind::Whitespace
        } else if is_alphabetic(c) {
            Kind::Alphabetic
        } else if jc::is_digit(u32::from(c)) {
            Kind::Numeric
        } else {
            Kind::Other
        };
        if state == Kind::Whitespace {
            if kind != Kind::Whitespace {
                start = ci;
            }
        } else if kind != state || (kind == Kind::Other && c != pc) {
            tokens.push(s[start..ci].to_vec());
            start = ci;
        }
        state = kind;
        pc = c;
    }
    if kind != Kind::Whitespace {
        tokens.push(s[start..].to_vec());
    }
    tokens
}

/// `CharacterNgramFeatureGenerator`'s `NGramModel.add(token, min, max)`:
/// the lower-cased token's distinct grams in insertion order (a length
/// past `Integer.MAX_VALUE - 1` overflows Java's loop bound: none).
// ARITH: gram offsets within the token; lengths checked against it.
#[allow(clippy::arithmetic_side_effects)]
fn character_ngrams(token: &[u16], min: i32, max: i32) -> Result<Vec<JString>, AnalysisError> {
    let lower = text::to_lower_case(token);
    let mut grams: Vec<JString> = Vec::new();
    let Some(end) = max.checked_add(1) else {
        return Ok(grams);
    };
    let mut seen = HashSet::new();
    let n = token.len();
    let top = i64::from(end).min(n as i64 + 1);
    for len in i64::from(min)..top {
        if len < 0 {
            // `subSequence(0, len)`
            return Err(AnalysisError::IllegalState(format!(
                "IndexOutOfBoundsException: subSequence(0, {len})"
            )));
        }
        let len = len as usize;
        let mut at = 0;
        while at + len <= n {
            let gram = lower.get(at..at + len).ok_or_else(|| {
                AnalysisError::IllegalState(
                    "BufferOverflowException: the lower-cased token is longer".into(),
                )
            })?;
            if seen.insert(gram) {
                grams.push(gram.to_vec());
            }
            at += 1;
        }
    }
    Ok(grams)
}

/// The previous outcomes a generator reads (`previousOutcomes`): as strings,
/// or as a beam sequence's outcome ids over the model's outcome names --
/// read by id, so a beam step need not copy every earlier outcome out.
#[derive(Debug, Clone, Copy)]
pub enum Prior<'a> {
    /// The outcomes themselves.
    Strings(&'a [JString]),
    /// Outcome ids into `outcomes`.
    Ids {
        /// The sequence's outcome ids.
        ids: &'a [u32],
        /// The model's outcome names.
        outcomes: &'a [JString],
    },
}

impl<'a> Prior<'a> {
    /// `previousOutcomes[i]`, if there is one.
    pub fn get(self, i: usize) -> Option<&'a [u16]> {
        match self {
            Prior::Strings(s) => s.get(i).map(Vec::as_slice),
            Prior::Ids { ids, outcomes } => ids
                .get(i)
                .and_then(|&o| outcomes.get(o as usize))
                .map(Vec::as_slice),
        }
    }

    /// `previousOutcomes.length`.
    pub fn len(self) -> usize {
        match self {
            Prior::Strings(s) => s.len(),
            Prior::Ids { ids, .. } => ids.len(),
        }
    }

    /// No previous outcome.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

impl Generator {
    /// [`Self::create_features_with`] over the previous outcomes as
    /// strings.
    pub fn create_features(
        &mut self,
        features: &mut Vec<JString>,
        tokens: &[JString],
        index: usize,
        prior: &[JString],
        sentence: u64,
    ) -> Result<(), AnalysisError> {
        self.create_features_with(features, tokens, index, Prior::Strings(prior), sentence)
    }

    /// `createFeatures(features, tokens, index, previousOutcomes)`;
    /// `sentence` identifies `tokens` (Java compares the array by
    /// identity). Fails where Java indexes past the previous outcomes.
    // ARITH: token indices and lengths of in-memory strings; window offsets are checked against the sentence.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn create_features_with(
        &mut self,
        features: &mut Vec<JString>,
        tokens: &[JString],
        index: usize,
        prior: Prior<'_>,
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
                inner.create_features_with(features, tokens, index, prior, sentence)?;
                let mut window = Vec::new();
                // Java tries every offset of the window; past the
                // sentence's ends none adds a feature, so the loops stop
                // there (a window of `Integer.MAX_VALUE` costs nothing)
                for i in 1..=((*prev).max(0) as usize).min(index) {
                    window.clear();
                    inner.create_features_with(&mut window, tokens, index - i, prior, sentence)?;
                    let p = format!("p{i}");
                    features.extend(window.iter().map(|f| cat(&p, f)));
                }
                let ahead = tokens.len().saturating_sub(index + 1);
                for i in 1..=((*next).max(0) as usize).min(ahead) {
                    window.clear();
                    inner.create_features_with(&mut window, tokens, index + i, prior, sentence)?;
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
                features.push(text::units2("wc=", class));
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
            Generator::PreviousMap(map) => {
                features.push(cat("pd=", &or_null(map.get(&tokens[index]))));
            }
            Generator::BigramName => {
                let wc = text::units(token_feature(&tokens[index]));
                if index > 0 {
                    features.push(comma("pw,w=", &[&tokens[index - 1], &tokens[index]]));
                    let pwc = text::units(token_feature(&tokens[index - 1]));
                    features.push(comma("pwc,wc=", &[&pwc, &wc]));
                }
                if index + 1 < tokens.len() {
                    features.push(comma("w,nw=", &[&tokens[index], &tokens[index + 1]]));
                    let nwc = text::units(token_feature(&tokens[index + 1]));
                    features.push(comma("wc,nc=", &[&wc, &nwc]));
                }
            }
            Generator::TrigramName => {
                let class = |i: usize| text::units(token_feature(&tokens[i]));
                let wc = class(index);
                if index > 1 {
                    features.push(comma(
                        "ppw,pw,w=",
                        &[&tokens[index - 2], &tokens[index - 1], &tokens[index]],
                    ));
                    features.push(comma(
                        "ppwc,pwc,wc=",
                        &[&class(index - 2), &class(index - 1), &wc],
                    ));
                }
                if index + 2 < tokens.len() {
                    features.push(comma(
                        "w,nw,nnw=",
                        &[&tokens[index], &tokens[index + 1], &tokens[index + 2]],
                    ));
                    features.push(comma(
                        "wc,nwc,nnwc=",
                        &[&wc, &class(index + 1), &class(index + 2)],
                    ));
                }
            }
            Generator::DocumentBegin(first) => {
                let first = *first.get_or_insert(sentence);
                if first == sentence && index == 0 {
                    features.push(text::units("D=begin"));
                }
            }
            Generator::CharacterNgram { min, max } => {
                for gram in character_ngrams(&tokens[index], *min, *max)? {
                    features.push(cat("ng=", &text::to_lower_case(&gram)));
                }
            }
            Generator::TokenPattern => {
                let token = &tokens[index];
                let tokenized = simple_tokenize(token);
                if tokenized.len() == 1 {
                    features.push(cat("st=", &text::to_lower_case(token)));
                    return Ok(());
                }
                features.push(cat("stn=", &text::int_units(tokenized.len())));
                let classes: Vec<&str> = tokenized.iter().map(|t| token_feature(t)).collect();
                let mut pattern = String::new();
                for (i, t) in tokenized.iter().enumerate() {
                    if i + 1 < tokenized.len() {
                        features.push(text::units(&format!(
                            "pt2={}{}",
                            classes[i],
                            classes[i + 1]
                        )));
                    }
                    if i + 2 < tokenized.len() {
                        features.push(text::units(&format!(
                            "pt3={}{}{}",
                            classes[i],
                            classes[i + 1],
                            classes[i + 2]
                        )));
                    }
                    pattern.push_str(classes[i]);
                    // `[^a-zA-Z]` found nowhere: letters only
                    if t.iter()
                        .all(|c| u8::try_from(*c).is_ok_and(|b| b.is_ascii_alphabetic()))
                    {
                        features.push(cat("st=", &text::to_lower_case(t)));
                    }
                }
                features.push(text::units(&format!("pta={pattern}")));
            }
            Generator::Dictionary {
                dictionary,
                sentence: current,
                names,
            } => {
                if *current != Some(sentence) {
                    *current = Some(sentence);
                    *names = dictionary_names(dictionary, tokens);
                }
                if names.iter().any(|&(s, e)| s <= index && index < e) {
                    features.push(text::units(":w=dic"));
                    features.push(cat(":w=dic=", &tokens[index]));
                }
            }
            Generator::WordCluster {
                clusters,
                name,
                lowercase,
            } => {
                let id = if *lowercase {
                    clusters.lookup(&text::to_lower_case(&tokens[index]))
                } else {
                    clusters.lookup(&tokens[index])
                };
                if let Some(id) = id {
                    features.push(text::concat(&[name, id]));
                }
            }
            Generator::BrownToken(clusters) => {
                for class in word_classes(&tokens[index], clusters) {
                    features.push(cat("browncluster=", &class));
                }
            }
            Generator::BrownTokenClass(clusters) => {
                let shape = token_feature(&tokens[index]);
                for class in word_classes(&tokens[index], clusters) {
                    features.push(cat(&format!("c,browncluster={shape},"), &class));
                }
            }
            Generator::BrownBigram(clusters) => {
                let wc = word_classes(&tokens[index], clusters);
                if index > 0 {
                    let prev = word_classes(&tokens[index - 1], clusters);
                    for (p, w) in prev.iter().zip(&wc) {
                        features.push(comma("pbrowncluster,browncluster=", &[p, w]));
                    }
                }
                if index + 1 < tokens.len() {
                    let next = word_classes(&tokens[index + 1], clusters);
                    for (w, n) in wc.iter().zip(&next) {
                        features.push(comma("browncluster,nbrowncluster=", &[w, n]));
                    }
                }
            }
            Generator::PosTaggerName { tagger, cached } => {
                let hit = cached.as_ref().is_some_and(|(t, _)| t == tokens);
                if !hit {
                    let tags = tagger.tag(tokens)?;
                    *cached = Some((tokens.to_vec(), tags));
                }
                if let Some((_, tags)) = cached {
                    features.push(cat("pos=", &tags[index]));
                }
            }
            Generator::Aggregated(gens) => {
                for g in gens {
                    g.create_features_with(features, tokens, index, prior, sentence)?;
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
                inner.create_features_with(&mut computed, tokens, index, prior, sentence)?;
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

    /// `updateAdaptiveData(tokens, outcomes)`: the previous-map generators
    /// learn each token's outcome; wrappers pass it on.
    pub fn update_adaptive_data(&mut self, tokens: &[JString], outcomes: &[JString]) {
        match self {
            Generator::PreviousMap(map) => {
                for (t, o) in tokens.iter().zip(outcomes) {
                    map.insert(t.clone(), o.clone());
                }
            }
            Generator::Window { inner, .. } | Generator::Cached(inner, _) => {
                inner.update_adaptive_data(tokens, outcomes);
            }
            Generator::Aggregated(gens) => {
                for g in gens {
                    g.update_adaptive_data(tokens, outcomes);
                }
            }
            _ => {}
        }
    }

    /// `clearAdaptiveData()`: the previous maps emptied, the document
    /// begun again; wrappers pass it on (the cache keeps its sentence, as
    /// Java's).
    pub fn clear_adaptive_data(&mut self) {
        match self {
            Generator::PreviousMap(map) => map.clear(),
            Generator::DocumentBegin(first) => *first = None,
            Generator::Window { inner, .. } | Generator::Cached(inner, _) => {
                inner.clear_adaptive_data();
            }
            Generator::Aggregated(gens) => {
                for g in gens {
                    g.clear_adaptive_data();
                }
            }
            _ => {}
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_views_agree() {
        let outcomes = vec![text::units("O"), text::units("B-x")];
        let strings = vec![text::units("B-x"), text::units("O")];
        let ids = Prior::Ids {
            ids: &[1, 0],
            outcomes: &outcomes,
        };
        let by_string = Prior::Strings(&strings);
        for p in [ids, by_string] {
            assert_eq!(p.len(), 2);
            assert!(!p.is_empty());
            assert_eq!(p.get(0), Some(&strings[0][..]));
            assert_eq!(p.get(1), Some(&strings[1][..]));
            assert_eq!(p.get(2), None);
        }
        assert!(Prior::Strings(&[]).is_empty());
    }
}
