//! Constructor/configuration port of Java's `CompressingCodec` and its six
//! concrete test codecs. Selected stored-field and term-vector descriptors are
//! retained once, share the same compression-mode object, and bypass the
//! delegate. All nine other factories forward on every invocation.
//!
//! This partial port has no compression, file I/O, custom header/geometry writer,
//! default native component graph, SPI installation or IndexWriter routing.
//! In particular it does not substitute SimpleText for binary term vectors.
//! Rust differences: [`CodecKind`] selects Java's six subclasses; shared
//! immutable settings replace Java references, `Option` represents a nullable mode,
//! and typed delegate callbacks return `Result`. The default Lucene104 delegate
//! is only a routing descriptor. Static versus newly allocated mode identities,
//! Java validation order, display and random overloads are preserved.

use std::{fmt, rc::Rc};

use lucene_codecs::codec::DefaultLucene104Codec;
use lucene_util::java_random::JavaRandom;

/// Configuration identity only, not a native compressor/decompressor factory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    Fast,
    FastDecompression,
    HighCompression,
    Dummy,
    DeflateWithPreset,
    Lz4WithPreset,
}
impl fmt::Display for CompressionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Fast => "FAST",
            Self::FastDecompression => "FAST_DECOMPRESSION",
            Self::HighCompression => "HIGH_COMPRESSION",
            Self::Dummy => "DUMMY",
            Self::DeflateWithPreset => "BEST_COMPRESSION",
            Self::Lz4WithPreset => "BEST_SPEED",
        })
    }
}
static STATIC_MODES: [CompressionMode; 4] = [
    CompressionMode::Fast,
    CompressionMode::FastDecompression,
    CompressionMode::HighCompression,
    CompressionMode::Dummy,
];

/// Java's shared static modes and per-constructor preset mode instances.
#[derive(Debug, Clone)]
pub enum ModeHandle {
    Static(&'static CompressionMode),
    Owned(Rc<CompressionMode>),
}
impl ModeHandle {
    pub fn new(mode: CompressionMode) -> Self {
        match mode {
            CompressionMode::Fast => Self::Static(&STATIC_MODES[0]),
            CompressionMode::FastDecompression => Self::Static(&STATIC_MODES[1]),
            CompressionMode::HighCompression => Self::Static(&STATIC_MODES[2]),
            CompressionMode::Dummy => Self::Static(&STATIC_MODES[3]),
            _ => Self::Owned(Rc::new(mode)),
        }
    }
    pub fn mode(&self) -> &CompressionMode {
        match self {
            Self::Static(mode) => mode,
            Self::Owned(mode) => mode,
        }
    }
    /// Reference identity, rather than equality of the selected algorithm.
    pub fn same_instance(&self, other: &Self) -> bool {
        std::ptr::eq(self.mode(), other.mode())
    }
}
impl fmt::Display for ModeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mode())
    }
}

/// The six concrete classes, in `randomInstance` selector order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecKind {
    Fast,
    FastDecompression,
    HighCompression,
    Dummy,
    DeflateWithPreset,
    Lz4WithPreset,
}
impl CodecKind {
    pub const ALL: [Self; 6] = [
        Self::Fast,
        Self::FastDecompression,
        Self::HighCompression,
        Self::Dummy,
        Self::DeflateWithPreset,
        Self::Lz4WithPreset,
    ];

    pub fn class_name(self) -> &'static str {
        match self {
            Self::Fast => "FastCompressingCodec",
            Self::FastDecompression => "FastDecompressionCompressingCodec",
            Self::HighCompression => "HighCompressionCompressingCodec",
            Self::Dummy => "DummyCompressingCodec",
            Self::DeflateWithPreset => "DeflateWithPresetCompressingCodec",
            Self::Lz4WithPreset => "LZ4WithPresetCompressingCodec",
        }
    }
    fn format_name(self) -> &'static str {
        match self {
            Self::Fast => "FastCompressingStoredFieldsData",
            Self::FastDecompression => "FastDecompressionCompressingStoredFieldsData",
            Self::HighCompression => "HighCompressionCompressingStoredFieldsData",
            Self::Dummy => "DummyCompressingStoredFieldsData",
            Self::DeflateWithPreset => "DeflateWithPresetCompressingStoredFieldsData",
            Self::Lz4WithPreset => "LZ4WithPresetCompressingStoredFieldsData",
        }
    }
    fn suffix(self) -> &'static str {
        match self {
            Self::Fast => "FastCompressingStoredFields",
            Self::FastDecompression => "FastDecompressionCompressingStoredFields",
            Self::HighCompression => "HighCompressionCompressingStoredFields",
            Self::Dummy => "DummyCompressingStoredFields",
            // Pinned Java deliberately observed: LZ4 uses the Deflate suffix.
            Self::DeflateWithPreset | Self::Lz4WithPreset => {
                "DeflateWithPresetCompressingStoredFields"
            }
        }
    }
    fn mode(self) -> CompressionMode {
        match self {
            Self::Fast => CompressionMode::Fast,
            Self::FastDecompression => CompressionMode::FastDecompression,
            Self::HighCompression => CompressionMode::HighCompression,
            Self::Dummy => CompressionMode::Dummy,
            Self::DeflateWithPreset => CompressionMode::DeflateWithPreset,
            Self::Lz4WithPreset => CompressionMode::Lz4WithPreset,
        }
    }
    /// Java: each subclass's no-argument constructor.
    pub fn defaults(self) -> CompressingCodec {
        let (chunk, docs) = match self {
            Self::Fast | Self::Dummy => (1 << 14, 128),
            Self::FastDecompression => (1 << 14, 256),
            Self::HighCompression => (61440, 512),
            Self::DeflateWithPreset | Self::Lz4WithPreset => (1 << 18, 512),
        };
        self.configured(chunk, docs, false, 10)
            .expect("fixed Java defaults are valid")
    }
    /// Java: each subclass's four-argument constructor.
    pub fn configured(
        self,
        chunk: i32,
        docs: i32,
        suffix: bool,
        shift: i32,
    ) -> Result<CompressingCodec, InvalidConfiguration> {
        CompressingCodec::build(
            self.format_name(),
            if suffix { self.suffix() } else { "" },
            Some(ModeHandle::new(self.mode())),
            chunk,
            docs,
            shift,
            DefaultLucene104Codec,
            Some(self),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidConfiguration(pub String);
impl fmt::Display for InvalidConfiguration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for InvalidConfiguration {}

/// Shared immutable constructor settings of each selected Java format.
#[derive(Debug)]
pub struct FormatSettings {
    names: Names,
    pub compression_mode: Option<ModeHandle>,
    pub chunk_size: i32,
    pub max_docs_per_chunk: i32,
    /// Stored fields call this `blockShift`; term vectors call it `blockSize`.
    pub block_shift: i32,
}
#[derive(Debug)]
enum Names {
    Standard(CodecKind, bool),
    Custom(Box<(Box<str>, Box<str>)>),
}
impl FormatSettings {
    pub fn format_name(&self) -> &str {
        match &self.names {
            Names::Standard(kind, _) => kind.format_name(),
            Names::Custom(names) => &names.0,
        }
    }
    pub fn segment_suffix(&self) -> &str {
        match &self.names {
            Names::Standard(kind, true) => kind.suffix(),
            Names::Standard(_, false) => "",
            Names::Custom(names) => &names.1,
        }
    }
    fn kind(&self) -> Option<CodecKind> {
        match &self.names {
            Names::Standard(kind, _) => Some(*kind),
            Names::Custom(_) => None,
        }
    }
    fn mode(&self) -> ModeDisplay<'_> {
        ModeDisplay(self.compression_mode.as_ref())
    }
    /// `null` is accepted by both Java format constructors and their displays.
    pub fn mode_display(&self) -> String {
        self.mode().to_string()
    }
}
struct ModeDisplay<'a>(Option<&'a ModeHandle>);
impl fmt::Display for ModeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(mode) => fmt::Display::fmt(mode, f),
            None => f.write_str("null"),
        }
    }
}
/// Only constructor state; no `fieldsReader` or `fieldsWriter` implementation.
#[derive(Debug)]
pub struct SelectedStoredFieldsFormat(pub Rc<FormatSettings>);
/// Only constructor state; no `vectorsReader` or `vectorsWriter` implementation.
#[derive(Debug)]
pub struct SelectedTermVectorsFormat(pub Rc<FormatSettings>);
impl fmt::Display for SelectedStoredFieldsFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.0;
        write!(f, "Lucene90CompressingStoredFieldsFormat(compressionMode={}, chunkSize={}, maxDocsPerChunk={}, blockShift={})",
            s.mode(), s.chunk_size, s.max_docs_per_chunk, s.block_shift)
    }
}
impl fmt::Display for SelectedTermVectorsFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.0;
        write!(f, "Lucene90CompressingTermVectorsFormat(compressionMode={}, chunkSize={}, maxDocsPerChunk={}, blockSize={})",
            s.mode(), s.chunk_size, s.max_docs_per_chunk, s.block_shift)
    }
}

/// Java: the abstract base, retaining the two selected format objects once.
#[derive(Debug)]
pub struct CompressingCodec<D = DefaultLucene104Codec> {
    delegate: D,
    formats: Box<SelectedFormats>,
}
#[derive(Debug)]
struct SelectedFormats {
    stored: SelectedStoredFieldsFormat,
    vectors: SelectedTermVectorsFormat,
}
impl CompressingCodec {
    /// Java: base constructor with an empty suffix.
    pub fn new(
        name: &str,
        mode: Option<ModeHandle>,
        chunk: i32,
        docs: i32,
        shift: i32,
    ) -> Result<Self, InvalidConfiguration> {
        Self::with_suffix(name, "", mode, chunk, docs, shift)
    }
    /// Java: base constructor with an explicit suffix.
    pub fn with_suffix(
        name: &str,
        suffix: &str,
        mode: Option<ModeHandle>,
        chunk: i32,
        docs: i32,
        shift: i32,
    ) -> Result<Self, InvalidConfiguration> {
        Self::with_delegate(
            name,
            suffix,
            mode,
            chunk,
            docs,
            shift,
            DefaultLucene104Codec,
        )
    }
    /// Java: `randomInstance(random, chunkSize, maxDocsPerChunk, suffix, blockShift)`.
    /// The selector draw occurs before constructor validation, even on error.
    pub fn random_explicit(
        random: &mut impl RandomSource,
        chunk: i32,
        docs: i32,
        suffix: bool,
        shift: i32,
    ) -> Result<Self, InvalidConfiguration> {
        let kind = match random.next_int_bounded(6) {
            0 => CodecKind::Fast,
            1 => CodecKind::FastDecompression,
            2 => CodecKind::HighCompression,
            3 => CodecKind::Dummy,
            4 => CodecKind::DeflateWithPreset,
            5 => CodecKind::Lz4WithPreset,
            _ => unreachable!("RandomSource must return a value below its bound"),
        };
        kind.configured(chunk, docs, suffix, shift)
    }
    /// Java: `randomInstance(random)`; preserve every conditional draw.
    pub fn random_instance(random: &mut impl RandomSource) -> Result<Self, InvalidConfiguration> {
        let chunk = if random.next_boolean() {
            between(random, 10, 100)
        } else {
            between(random, 10, 1 << 15)
        };
        let docs = if random.next_boolean() {
            between(random, 1, 10)
        } else {
            between(random, 64, 1024)
        };
        let shift = if random.next_boolean() {
            between(random, 2, 10)
        } else {
            between(random, 2, 22)
        };
        Self::random_explicit(random, chunk, docs, false, shift)
    }
    /// Java: `reasonableInstance(random)`.
    pub fn reasonable_instance(
        random: &mut impl RandomSource,
    ) -> Result<Self, InvalidConfiguration> {
        let chunk = between(random, 1 << 13, 1 << 17);
        let docs = between(random, 1 << 6, 1 << 10);
        let shift = between(random, 8, 12);
        Self::random_explicit(random, chunk, docs, false, shift)
    }
    /// Java: `randomInstance(random, withSegmentSuffix)`. Its 1..1024 shift
    /// range usually fails the selected constructor; it must not be clamped.
    pub fn random_with_suffix(
        random: &mut impl RandomSource,
        suffix: bool,
    ) -> Result<Self, InvalidConfiguration> {
        let chunk = between(random, 1, 1 << 15);
        let docs = between(random, 64, 1024);
        let shift = between(random, 1, 1024);
        Self::random_explicit(random, chunk, docs, suffix, shift)
    }
}
impl<D> CompressingCodec<D> {
    /// Typed replacement for Java's retained FilterCodec delegate.
    #[allow(clippy::too_many_arguments)]
    pub fn with_delegate(
        name: &str,
        suffix: &str,
        mode: Option<ModeHandle>,
        chunk: i32,
        docs: i32,
        shift: i32,
        delegate: D,
    ) -> Result<Self, InvalidConfiguration> {
        Self::build(name, suffix, mode, chunk, docs, shift, delegate, None)
    }
    #[allow(clippy::too_many_arguments)]
    fn build(
        name: &str,
        suffix: &str,
        mode: Option<ModeHandle>,
        chunk: i32,
        docs: i32,
        shift: i32,
        delegate: D,
        kind: Option<CodecKind>,
    ) -> Result<Self, InvalidConfiguration> {
        // Java: Codec -> NamedSPILoader.checkServiceName, before either format.
        if name.encode_utf16().count() >= 128 {
            return Err(InvalidConfiguration(format!(
                "Illegal service name: '{name}' is too long (must be < 128 chars)."
            )));
        }
        if !name.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return Err(InvalidConfiguration(format!(
                "Illegal service name: '{name}' must be simple ascii alphanumeric."
            )));
        }
        // Java: stored-field validation occurs before term-vector creation.
        if chunk < 1 {
            return Err(InvalidConfiguration("chunkSize must be >= 1".into()));
        }
        if docs < 1 {
            return Err(InvalidConfiguration("maxDocsPerChunk must be >= 1".into()));
        }
        if !(2..=22).contains(&shift) {
            return Err(InvalidConfiguration(format!(
                "blockSize must be in 2-22, got {shift}"
            )));
        }
        let settings = Rc::new(FormatSettings {
            names: kind.map_or_else(
                || Names::Custom(Box::new((name.into(), suffix.into()))),
                |kind| Names::Standard(kind, !suffix.is_empty()),
            ),
            compression_mode: mode,
            chunk_size: chunk,
            max_docs_per_chunk: docs,
            block_shift: shift,
        });
        let stored = SelectedStoredFieldsFormat(settings.clone());
        // Every term-vector precondition follows from successful SF validation.
        let vectors = SelectedTermVectorsFormat(settings);
        Ok(Self {
            delegate,
            formats: Box::new(SelectedFormats { stored, vectors }),
        })
    }
    pub fn name(&self) -> &str {
        self.formats.stored.0.format_name()
    }
    pub fn kind(&self) -> Option<CodecKind> {
        self.formats.stored.0.kind()
    }
    pub fn delegate(&self) -> &D {
        &self.delegate
    }
    pub fn stored_fields_format(&self) -> &SelectedStoredFieldsFormat {
        &self.formats.stored
    }
    pub fn term_vectors_format(&self) -> &SelectedTermVectorsFormat {
        &self.formats.vectors
    }
    /// Retained shared settings node, including Rc counters/alignment, and the
    /// custom-name descriptor when present. String payload and mode allocation
    /// are separate; allocator metadata is excluded.
    pub fn shared_settings_bytes(&self) -> usize {
        let (layout, _) = std::alloc::Layout::new::<[usize; 2]>()
            .extend(std::alloc::Layout::new::<FormatSettings>())
            .expect("Rc settings layout");
        layout.pad_to_align().size()
            + match &self.formats.stored.0.names {
                Names::Standard(_, _) => 0,
                Names::Custom(_) => std::mem::size_of::<(Box<str>, Box<str>)>(),
            }
    }
    /// Owned name/suffix payload, retained once for custom base codecs.
    pub fn owned_string_bytes(&self) -> usize {
        match &self.formats.stored.0.names {
            Names::Standard(_, _) => 0,
            Names::Custom(names) => names.0.len() + names.1.len(),
        }
    }
    /// Replace the descriptor with an explicit typed callback implementation.
    /// Selected formats retain their values and identities.
    pub fn map_delegate<E>(self, map: impl FnOnce(D) -> E) -> CompressingCodec<E> {
        CompressingCodec {
            delegate: map(self.delegate),
            formats: self.formats,
        }
    }
}
impl<D> fmt::Display for CompressingCodec<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(storedFieldsFormat={}, termVectorsFormat={})",
            self.name(),
            self.formats.stored,
            self.formats.vectors
        )
    }
}

/// The nine inherited Java FilterCodec factories; SF/TV cannot enter this seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentKind {
    DocValues,
    FieldInfos,
    LiveDocs,
    Norms,
    Postings,
    SegmentInfo,
    Compound,
    Points,
    KnnVectors,
}
pub trait CodecDelegate {
    type Component;
    type Error;
    fn component(&self, kind: ComponentKind) -> Result<Self::Component, Self::Error>;
}
impl<D: CodecDelegate> CompressingCodec<D> {
    /// Java: inherited FilterCodec accessor, without caching or error conversion.
    pub fn component(&self, kind: ComponentKind) -> Result<D::Component, D::Error> {
        self.delegate.component(kind)
    }
}

/// Only Java Random operations used by these constructors. Implementations must
/// return a value in `0..bound`; the normal implementation is JavaRandom.
pub trait RandomSource {
    fn next_boolean(&mut self) -> bool;
    fn next_int_bounded(&mut self, bound: i32) -> i32;
}
impl RandomSource for JavaRandom {
    fn next_boolean(&mut self) -> bool {
        JavaRandom::next_boolean(self)
    }
    fn next_int_bounded(&mut self, bound: i32) -> i32 {
        JavaRandom::next_int_bounded(self, bound)
    }
}
fn between(random: &mut impl RandomSource, min: i32, max: i32) -> i32 {
    // Java RandomNumbers.randomIntBetween; all callers have fixed small bounds.
    min + random.next_int_bounded(max - min + 1)
}

#[cfg(test)]
mod tests;
