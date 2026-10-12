//! `BaseModel` and its five subclasses Lucene loads: a model package is a
//! zip of artifacts -- `manifest.properties`, the event model (`.model`),
//! dictionaries (`.dictionary`, `.tagdict`), the feature generator
//! descriptor (`.featuregen`) -- read in `BaseModel.loadModel`'s two passes
//! (the manifest first, which names the tool factory, then every entry
//! through the serializer its extension or its `serializer-class-<entry>`
//! property names) and checked by `validateArtifactMap`.
//!
//! An error while reading is the `IOException` Java throws there
//! (`InvalidFormatException`, `ZipException`, `EOFException`, ...: kind
//! `Io`, the class named in the message); one from validation is
//! `checkArtifactMap`'s `IllegalArgumentException` wrapping it.
//!
//! Refused (`UnsupportedOperationException`): a tool factory or artifact
//! serializer class other than OpenNLP's own for the component -- Java
//! loads a user's subclass by reflection, which a port cannot.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis::factory::{FactoryError, JavaException};

use super::dictionary::{Dictionary, PosDictionary};
use super::ml::MaxentModel;
use super::properties::Properties;
use super::text;
use super::zip::ZipReader;

const MANIFEST_ENTRY: &str = "manifest.properties";

/// The OpenNLP release whose runtime this is (`Version.currentVersion()`).
const CURRENT_VERSION: &str = "2.5.3";

/// The model packages Lucene's module loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    /// `SentenceModel`.
    SentenceDetector,
    /// `TokenizerModel`.
    Tokenizer,
    /// `POSModel`.
    PosTagger,
    /// `ChunkerModel`.
    Chunker,
    /// `LemmatizerModel`.
    Lemmatizer,
}

impl Component {
    /// Every component.
    const ALL: [Component; 5] = [
        Component::SentenceDetector,
        Component::Tokenizer,
        Component::PosTagger,
        Component::Chunker,
        Component::Lemmatizer,
    ];

    /// `COMPONENT_NAME`.
    fn name(self) -> &'static str {
        match self {
            Component::SentenceDetector => "SentenceDetectorME",
            Component::Tokenizer => "TokenizerME",
            Component::PosTagger => "POSTaggerME",
            Component::Chunker => "ChunkerME",
            Component::Lemmatizer => "StatisticalLemmatizer",
        }
    }

    /// `getDefaultFactory()`: OpenNLP's factory class for the component.
    fn factory_class(self) -> &'static str {
        match self {
            Component::SentenceDetector => "opennlp.tools.sentdetect.SentenceDetectorFactory",
            Component::Tokenizer => "opennlp.tools.tokenize.TokenizerFactory",
            Component::PosTagger => "opennlp.tools.postag.POSTaggerFactory",
            Component::Chunker => "opennlp.tools.chunker.ChunkerFactory",
            Component::Lemmatizer => "opennlp.tools.lemmatizer.LemmatizerFactory",
        }
    }

    /// The event model's entry.
    fn model_entry(self) -> &'static str {
        match self {
            Component::SentenceDetector => "sent.model",
            Component::Tokenizer => "token.model",
            Component::PosTagger => "pos.model",
            Component::Chunker => "chunker.model",
            Component::Lemmatizer => "lemmatizer.model",
        }
    }
}

/// A loaded artifact.
#[derive(Debug, Clone)]
pub enum Artifact {
    /// `Properties` (`.properties`).
    Properties(Properties),
    /// A `MaxentModel` (`.model`).
    Model(Arc<MaxentModel>),
    /// A `Dictionary` (`.dictionary`).
    Dictionary(Arc<Dictionary>),
    /// A `POSDictionary` (`.tagdict`).
    PosDictionary(Arc<PosDictionary>),
    /// Raw bytes (`.txt`, `.html`, `.featuregen`).
    Bytes(Arc<Vec<u8>>),
}

impl Artifact {
    /// `artifact.toString()`, as a validation message prints it.
    fn describe(&self) -> &'static str {
        match self {
            Artifact::Properties(_) => "Properties",
            Artifact::Model(_) => "MaxentModel",
            Artifact::Dictionary(_) => "Dictionary",
            Artifact::PosDictionary(_) => "POSDictionary",
            Artifact::Bytes(_) => "byte[]",
        }
    }
}

/// `opennlp.tools.util.Version`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    /// The major version.
    pub major: i32,
    /// The minor version.
    pub minor: i32,
    /// The revision.
    pub revision: i32,
    /// Whether it ends in `-SNAPSHOT`.
    pub snapshot: bool,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snapshot = if self.snapshot { "-SNAPSHOT" } else { "" };
        write!(
            f,
            "{}.{}.{}{snapshot}",
            self.major, self.minor, self.revision
        )
    }
}

impl Version {
    /// `Version.parse`: `major.minor.revision[-...]`; Java's
    /// `NumberFormatException` message on failure.
    // ARITH: indices from `find` inside the version string.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn parse(version: &str) -> Result<Version, String> {
        let first = version.find('.');
        let second = first.and_then(|f| version[f + 1..].find('.').map(|s| s + f + 1));
        let (Some(first), Some(second)) = (first, second) else {
            return Err(format!(
                "Invalid version format '{version}', expected two dots!"
            ));
        };
        let end = version.find('-').unwrap_or(version.len());
        let int = |s: &str| {
            text::parse_int(&text::units(s)).ok_or_else(|| format!("For input string: \"{s}\""))
        };
        let revision = version
            .get(second + 1..end)
            .ok_or_else(|| format!("begin {}, end {end}, length {}", second + 1, version.len()))?;
        Ok(Version {
            major: int(&version[..first])?,
            minor: int(&version[first + 1..second])?,
            revision: int(revision)?,
            snapshot: version.ends_with("-SNAPSHOT"),
        })
    }
}

/// A read model package: its manifest and its artifacts.
#[derive(Debug, Clone)]
pub struct BaseModel {
    component: Component,
    manifest: Properties,
    artifacts: HashMap<String, Artifact>,
    version: Version,
    maxent: Arc<MaxentModel>,
}

fn invalid_format(message: impl Into<String>) -> FactoryError {
    FactoryError::new(
        JavaException::Io,
        format!("InvalidFormatException: {}", message.into()),
    )
}

/// `checkArtifactMap`'s `new IllegalArgumentException(e)` around an
/// `InvalidFormatException` from validation.
fn invalid(message: impl Into<String>) -> FactoryError {
    FactoryError::illegal_argument(format!(
        "opennlp.tools.util.InvalidFormatException: {}",
        message.into()
    ))
}

fn unsupported(message: impl Into<String>) -> FactoryError {
    FactoryError::new(JavaException::UnsupportedOperation, message)
}

/// `getEntryExtension`: what follows the last `.` (the whole name when it
/// has none).
// ARITH: `rfind` gives an index inside the name.
#[allow(clippy::arithmetic_side_effects)]
fn extension(entry: &str) -> Result<&str, FactoryError> {
    let i = entry.rfind('.').map_or(0, |i| i + 1);
    if i >= entry.len() {
        return Err(invalid_format(format!(
            "Entry name must have type extension: {entry}"
        )));
    }
    Ok(&entry[i..])
}

/// The serializers a component's model registers: `BaseModel`'s
/// (`model`, `properties`, `dictionary`, `txt`, `html`) plus the POS
/// tagger factory's `tagdict` (by the manifest's factory) and
/// `POSModel`'s `featuregen` (by the model class).
fn read_artifact(
    component: Component,
    factory: Component,
    ext: &str,
    data: Vec<u8>,
) -> Result<Artifact, FactoryError> {
    Ok(match ext {
        "model" => Artifact::Model(Arc::new(MaxentModel::read(&data)?)),
        "properties" => Artifact::Properties(Properties::load(&data)?),
        "dictionary" => Artifact::Dictionary(Arc::new(Dictionary::read(&data)?)),
        "txt" | "html" => Artifact::Bytes(Arc::new(data)),
        "tagdict" if factory == Component::PosTagger => {
            Artifact::PosDictionary(Arc::new(PosDictionary::read(&data)?))
        }
        "featuregen" if component == Component::PosTagger => Artifact::Bytes(Arc::new(data)),
        other => return Err(invalid_format(format!("Unknown artifact format: {other}"))),
    })
}

/// An artifact through the serializer class a manifest names for it.
fn read_with_serializer(class: &str, data: Vec<u8>) -> Result<Artifact, FactoryError> {
    Ok(match class {
        "opennlp.tools.util.model.DictionarySerializer" => {
            Artifact::Dictionary(Arc::new(Dictionary::read(&data)?))
        }
        "opennlp.tools.postag.POSTaggerFactory$POSDictionarySerializer" => {
            Artifact::PosDictionary(Arc::new(PosDictionary::read(&data)?))
        }
        "opennlp.tools.util.model.GenericModelSerializer" => {
            Artifact::Model(Arc::new(MaxentModel::read(&data)?))
        }
        "opennlp.tools.util.model.PropertiesSerializer" => {
            Artifact::Properties(Properties::load(&data)?)
        }
        "opennlp.tools.util.model.ByteArraySerializer" => Artifact::Bytes(Arc::new(data)),
        other => {
            return Err(unsupported(format!(
                "the artifact serializer {other} is not supported by this port"
            )))
        }
    })
}

/// `ModelUtil.validateOutcomes`: exactly the expected outcomes.
fn has_outcomes(model: &MaxentModel, expected: &[&str]) -> bool {
    model.num_outcomes() == expected.len()
        && (0..model.num_outcomes()).all(|i| expected.contains(&model.outcome_name(i)))
}

impl BaseModel {
    /// `new XModel(InputStream)`: the package read and validated as a
    /// `component` model.
    pub fn load(component: Component, bytes: &[u8]) -> Result<BaseModel, FactoryError> {
        // first pass: the manifest
        let mut zip = ZipReader::new(bytes);
        let mut manifest = None;
        while let Some(entry) = zip.next_entry()? {
            if entry.name == MANIFEST_ENTRY {
                manifest = Some(Properties::load(&entry.data)?);
                break;
            }
        }
        let first_manifest = manifest.ok_or_else(|| {
            FactoryError::new(
                JavaException::NullPointer,
                "the model package has no manifest.properties",
            )
        })?;
        // initializeFactory: the factory the manifest names (one of
        // OpenNLP's five, possibly another component's), else the default
        let factory = match first_manifest.get("factory") {
            None => component,
            Some(name) => Component::ALL
                .into_iter()
                .find(|c| c.factory_class() == name)
                .ok_or_else(|| {
                    unsupported(format!(
                        "the tool factory {name} is not supported by this port"
                    ))
                })?,
        };
        // second pass: every artifact
        let mut artifacts = HashMap::new();
        let mut zip = ZipReader::new(bytes);
        while let Some(entry) = zip.next_entry()? {
            let ext = extension(&entry.name)?.to_string();
            let class = first_manifest.get(&format!("serializer-class-{}", entry.name));
            let artifact = match class {
                Some(class) => read_with_serializer(class, entry.data)?,
                None => read_artifact(component, factory, &ext, entry.data)?,
            };
            artifacts.insert(entry.name, artifact);
        }
        let manifest = match artifacts.get(MANIFEST_ENTRY) {
            Some(Artifact::Properties(p)) => p.clone(),
            Some(_) => return Err(invalid(format!("Missing the {MANIFEST_ENTRY}!"))),
            None => first_manifest,
        };
        let version = Self::validate_base(component, &manifest)?;
        Self::validate_factory(factory, &manifest, &artifacts)?;
        let maxent = Self::validate_model(component, &artifacts)?;
        if factory != component {
            // Java's tool constructor casts the factory: a
            // ClassCastException when it creates the tool
            return Err(unsupported(format!(
                "a {} model with the tool factory {}",
                component.name(),
                factory.factory_class()
            )));
        }
        Ok(BaseModel {
            component,
            manifest,
            artifacts,
            version,
            maxent,
        })
    }

    /// `BaseModel.validateArtifactMap`.
    fn validate_base(component: Component, manifest: &Properties) -> Result<Version, FactoryError> {
        let version_string = manifest.get("OpenNLP-Version").ok_or_else(|| {
            invalid(format!(
                "Missing OpenNLP-Version property in {MANIFEST_ENTRY}!"
            ))
        })?;
        let version = Version::parse(version_string)
            .map_err(|_| invalid(format!("Unable to parse model version '{version_string}'!")))?;
        if version.major != 1 && version.major != 2 {
            return Err(invalid(format!(
                "Model version {version} is not supported by this ({CURRENT_VERSION}) version of OpenNLP!"
            )));
        }
        if version.snapshot {
            return Err(invalid(format!(
                "Model version {version} is a snapshot - snapshot models are not supported by this non-snapshot version ({CURRENT_VERSION}) of OpenNLP!"
            )));
        }
        let name = manifest.get("Component-Name").ok_or_else(|| {
            invalid(format!(
                "Missing Component-Name property in {MANIFEST_ENTRY}!"
            ))
        })?;
        if name != component.name() {
            return Err(invalid(format!(
                "The {} cannot load a model for the {name}!",
                component.name()
            )));
        }
        if manifest.get("Language").is_none() {
            return Err(invalid(format!(
                "Missing Language property in {MANIFEST_ENTRY}!"
            )));
        }
        Ok(version)
    }

    /// The tool factory's `validateArtifactMap`.
    fn validate_factory(
        component: Component,
        manifest: &Properties,
        artifacts: &HashMap<String, Artifact>,
    ) -> Result<(), FactoryError> {
        let wrong_dictionary = |entry: &str| match artifacts.get(entry) {
            Some(
                a @ (Artifact::Properties(_)
                | Artifact::Model(_)
                | Artifact::PosDictionary(_)
                | Artifact::Bytes(_)),
            ) => Err(a.describe()),
            _ => Ok(()),
        };
        match component {
            Component::SentenceDetector | Component::Tokenizer => {
                let mandatory = if component == Component::Tokenizer {
                    "useAlphaNumericOptimization"
                } else {
                    "useTokenEnd"
                };
                if manifest.get(mandatory).is_none() {
                    return Err(invalid(format!("{mandatory} is a mandatory property!")));
                }
                wrong_dictionary("abbreviations.dictionary").map_err(|a| {
                    invalid(format!(
                        "Abbreviations dictionary '{a}' has wrong type, needs to be of type Dictionary!"
                    ))
                })
            }
            Component::PosTagger => {
                if let Some(a) = artifacts.get("tags.tagdict") {
                    if !matches!(a, Artifact::PosDictionary(_)) {
                        return Err(invalid("POSTag dictionary has wrong type!"));
                    }
                }
                wrong_dictionary("ngram.dictionary")
                    .map_err(|_| invalid("NGram dictionary has wrong type!"))
            }
            Component::Chunker | Component::Lemmatizer => Ok(()),
        }
    }

    /// The model subclass's `validateArtifactMap`.
    fn validate_model(
        component: Component,
        artifacts: &HashMap<String, Artifact>,
    ) -> Result<Arc<MaxentModel>, FactoryError> {
        let model = match artifacts.get(component.model_entry()) {
            Some(Artifact::Model(m)) => Some(m),
            _ => None,
        };
        let m = match component {
            Component::SentenceDetector => {
                let m = model.ok_or_else(|| invalid("Unable to find sent.model maxent model!"))?;
                if !has_outcomes(m, &["s", "n"]) {
                    return Err(invalid(
                        "The maxent model is not compatible with the sentence detector!",
                    ));
                }
                m
            }
            Component::Tokenizer => {
                let m = model.ok_or_else(|| invalid("Token model is incomplete!"))?;
                if !has_outcomes(m, &["T", "F"]) {
                    return Err(invalid(
                        "The maxent model is not compatible with the tokenizer!",
                    ));
                }
                m
            }
            Component::PosTagger => model.ok_or_else(|| invalid("POS model is incomplete!"))?,
            Component::Chunker => model.ok_or_else(|| invalid("Chunker model is incomplete!"))?,
            Component::Lemmatizer => {
                model.ok_or_else(|| invalid("Lemmatizer model is incomplete!"))?
            }
        };
        Ok(Arc::clone(m))
    }

    /// `getManifestProperty(key)`.
    pub fn manifest_property(&self, key: &str) -> Option<&str> {
        self.manifest.get(key)
    }

    /// The manifest's `OpenNLP-Version`.
    pub fn version(&self) -> Version {
        self.version
    }

    /// The component's event model (validated present).
    pub fn maxent_model(&self) -> Arc<MaxentModel> {
        Arc::clone(&self.maxent)
    }

    /// The component the package was read as.
    pub fn component(&self) -> Component {
        self.component
    }

    /// The `Dictionary` artifact `entry`, if it is one.
    pub fn dictionary(&self, entry: &str) -> Option<Arc<Dictionary>> {
        match self.artifacts.get(entry) {
            Some(Artifact::Dictionary(d)) => Some(Arc::clone(d)),
            _ => None,
        }
    }

    /// The `POSDictionary` artifact `entry`, if it is one.
    pub fn pos_dictionary(&self, entry: &str) -> Option<Arc<PosDictionary>> {
        match self.artifacts.get(entry) {
            Some(Artifact::PosDictionary(d)) => Some(Arc::clone(d)),
            _ => None,
        }
    }

    /// The byte-array artifact `entry`, if it is one.
    pub fn bytes(&self, entry: &str) -> Option<Arc<Vec<u8>>> {
        match self.artifacts.get(entry) {
            Some(Artifact::Bytes(b)) => Some(Arc::clone(b)),
            _ => None,
        }
    }

    /// `BeamSize` from the manifest, else `default`: Java's
    /// `Integer.parseInt` (a `NumberFormatException` for a bad one).
    pub fn beam_size(&self, default: usize) -> Result<usize, lucene_analysis::AnalysisError> {
        match self.manifest_property("BeamSize") {
            None => Ok(default),
            Some(s) => match text::parse_int(&text::units(s)) {
                // a beam of size 0 or less keeps no sequence (Java: none of
                // its loops run), as size 0 here
                Some(n) => Ok(usize::try_from(n).unwrap_or(0)),
                None => Err(lucene_analysis::AnalysisError::IllegalArgument(format!(
                    "NumberFormatException: For input string: \"{s}\""
                ))),
            },
        }
    }
}
