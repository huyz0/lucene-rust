//! `org.apache.lucene.util.ResourceLoader`, `ClasspathResourceLoader`,
//! `org.apache.lucene.analysis.util.FilesystemResourceLoader`, and the
//! word-file helpers of `AbstractAnalysisFactory` (`getWordSet`,
//! `getLines`, `getSnowballWordSet`).
//!
//! Java's loader also loads classes (`findClass`, `newInstance`); a Rust
//! class cannot be found by name at run time, so the classes configuration
//! may name live in [`super::spi`]'s table instead, the same for every
//! loader.
//!
//! [`ClasspathResourceLoader`] serves the resource files this crate vendors
//! from the analysis-common jar, at their jar paths
//! (`org/apache/lucene/analysis/snowball/german_stop.txt`): the 40 stopword
//! and RSLP files the analyzers use, and `snowball/english_stop.txt` and
//! `cjk/stopwords.txt`, which configurations name. The jar's
//! `hyphenation.dtd` is not vendored (the grammar parser skips the
//! `DOCTYPE`), so the loader reports it as not found.
//!
//! Resource bytes are decoded as UTF-8 with Java's `CodingErrorAction.REPORT`:
//! a malformed sequence is a `MalformedInputException` (`Input length =
//! n`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::args::split_file_names;
use super::{FactoryError, JavaException};
use crate::{wordlist_loader, CharArraySet};

/// `org.apache.lucene.util.ResourceLoader`, resources only.
pub trait ResourceLoader: Send + Sync {
    /// `openResource(String)`: the resource's bytes.
    fn open_resource(&self, resource: &str) -> Result<Vec<u8>, FactoryError>;
}

macro_rules! vendored {
    ($($jar:literal => $file:literal),* $(,)?) => {
        &[$(($jar, include_bytes!(concat!("../lang/stopwords/", $file)))),*]
    };
}

/// The jar resources this crate vendors, by jar path.
const VENDORED: &[(&str, &[u8])] = vendored! {
    "org/apache/lucene/analysis/ar/stopwords.txt" => "ar_stopwords.txt",
    "org/apache/lucene/analysis/bg/stopwords.txt" => "bg_stopwords.txt",
    "org/apache/lucene/analysis/bn/stopwords.txt" => "bn_stopwords.txt",
    "org/apache/lucene/analysis/br/stopwords.txt" => "br_stopwords.txt",
    "org/apache/lucene/analysis/ca/stopwords.txt" => "ca_stopwords.txt",
    "org/apache/lucene/analysis/ckb/stopwords.txt" => "ckb_stopwords.txt",
    "org/apache/lucene/analysis/cz/stopwords.txt" => "cz_stopwords.txt",
    "org/apache/lucene/analysis/el/stopwords.txt" => "el_stopwords.txt",
    "org/apache/lucene/analysis/et/stopwords.txt" => "et_stopwords.txt",
    "org/apache/lucene/analysis/eu/stopwords.txt" => "eu_stopwords.txt",
    "org/apache/lucene/analysis/fa/stopwords.txt" => "fa_stopwords.txt",
    "org/apache/lucene/analysis/gl/stopwords.txt" => "gl_stopwords.txt",
    "org/apache/lucene/analysis/gl/galician.rslp" => "galician.rslp",
    "org/apache/lucene/analysis/hi/stopwords.txt" => "hi_stopwords.txt",
    "org/apache/lucene/analysis/hy/stopwords.txt" => "hy_stopwords.txt",
    "org/apache/lucene/analysis/id/stopwords.txt" => "id_stopwords.txt",
    "org/apache/lucene/analysis/lt/stopwords.txt" => "lt_stopwords.txt",
    "org/apache/lucene/analysis/lv/stopwords.txt" => "lv_stopwords.txt",
    "org/apache/lucene/analysis/ne/stopwords.txt" => "ne_stopwords.txt",
    "org/apache/lucene/analysis/pt/portuguese.rslp" => "portuguese.rslp",
    "org/apache/lucene/analysis/ro/stopwords.txt" => "ro_stopwords.txt",
    "org/apache/lucene/analysis/sr/stopwords.txt" => "sr_stopwords.txt",
    "org/apache/lucene/analysis/ta/stopwords.txt" => "ta_stopwords.txt",
    "org/apache/lucene/analysis/te/stopwords.txt" => "te_stopwords.txt",
    "org/apache/lucene/analysis/th/stopwords.txt" => "th_stopwords.txt",
    "org/apache/lucene/analysis/tr/stopwords.txt" => "tr_stopwords.txt",
    "org/apache/lucene/analysis/cjk/stopwords.txt" => "cjk_stopwords.txt",
    "org/apache/lucene/analysis/snowball/danish_stop.txt" => "danish_stop.txt",
    "org/apache/lucene/analysis/snowball/english_stop.txt" => "english_stop.txt",
    "org/apache/lucene/analysis/snowball/dutch_stop.txt" => "dutch_stop.txt",
    "org/apache/lucene/analysis/snowball/finnish_stop.txt" => "finnish_stop.txt",
    "org/apache/lucene/analysis/snowball/french_stop.txt" => "french_stop.txt",
    "org/apache/lucene/analysis/snowball/german_stop.txt" => "german_stop.txt",
    "org/apache/lucene/analysis/snowball/hungarian_stop.txt" => "hungarian_stop.txt",
    "org/apache/lucene/analysis/snowball/indonesian_stop.txt" => "indonesian_stop.txt",
    "org/apache/lucene/analysis/snowball/irish_stop.txt" => "irish_stop.txt",
    "org/apache/lucene/analysis/snowball/italian_stop.txt" => "italian_stop.txt",
    "org/apache/lucene/analysis/snowball/norwegian_stop.txt" => "norwegian_stop.txt",
    "org/apache/lucene/analysis/snowball/portuguese_stop.txt" => "portuguese_stop.txt",
    "org/apache/lucene/analysis/snowball/russian_stop.txt" => "russian_stop.txt",
    "org/apache/lucene/analysis/snowball/spanish_stop.txt" => "spanish_stop.txt",
    "org/apache/lucene/analysis/snowball/swedish_stop.txt" => "swedish_stop.txt",
};

/// `org.apache.lucene.util.ClasspathResourceLoader` over the vendored jar
/// resources (see the module docs).
#[derive(Debug, Clone, Copy, Default)]
pub struct ClasspathResourceLoader;

impl ClasspathResourceLoader {
    /// The jar paths it serves.
    pub fn resources() -> impl Iterator<Item = &'static str> {
        VENDORED.iter().map(|(path, _)| *path)
    }
}

impl ResourceLoader for ClasspathResourceLoader {
    // Java: ClasspathResourceLoader.openResource
    fn open_resource(&self, resource: &str) -> Result<Vec<u8>, FactoryError> {
        VENDORED
            .iter()
            .find(|(path, _)| *path == resource)
            .map(|(_, bytes)| bytes.to_vec())
            .ok_or_else(|| {
                FactoryError::io(format!(
                    "Resource not found (if you use Java Module System, make sure to open \
                     module and package containing resources to 'org.apache.lucene.core' \
                     module): {resource}"
                ))
            })
    }
}

/// `org.apache.lucene.analysis.util.FilesystemResourceLoader`: resources
/// relative to a directory, falling back to a delegate for a file that does
/// not exist there.
pub struct FilesystemResourceLoader {
    base_directory: PathBuf,
    delegate: Box<dyn ResourceLoader>,
}

impl FilesystemResourceLoader {
    /// `FilesystemResourceLoader(Path baseDirectory, ResourceLoader
    /// delegate)`: `IllegalArgumentException` unless the base is a
    /// directory.
    pub fn new(
        base_directory: impl Into<PathBuf>,
        delegate: Box<dyn ResourceLoader>,
    ) -> Result<Self, FactoryError> {
        let base_directory = base_directory.into();
        if !base_directory.is_dir() {
            return Err(FactoryError::illegal_argument(format!(
                "{} is not a directory",
                base_directory.display()
            )));
        }
        Ok(FilesystemResourceLoader {
            base_directory,
            delegate,
        })
    }

    /// `FilesystemResourceLoader(Path, ClassLoader)`: the delegate is the
    /// [`ClasspathResourceLoader`].
    pub fn with_classpath(base_directory: impl Into<PathBuf>) -> Result<Self, FactoryError> {
        Self::new(base_directory, Box::new(ClasspathResourceLoader))
    }

    /// The base directory.
    pub fn base_directory(&self) -> &Path {
        &self.base_directory
    }
}

impl ResourceLoader for FilesystemResourceLoader {
    // Java: FilesystemResourceLoader.openResource
    fn open_resource(&self, resource: &str) -> Result<Vec<u8>, FactoryError> {
        let path = self.base_directory.join(resource);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.delegate.open_resource(resource)
            }
            Err(e) => Err(io_error_message(&e, resource)),
        }
    }
}

/// The `IOException` Java's `Files.newInputStream(path)` and its first
/// read report: a directory is `"Is a directory"`, an unreadable file an
/// `AccessDeniedException` whose message is the path (the resource name
/// here, never the base directory's absolute path), anything else the
/// system's reason without Rust's `(os error n)` suffix.
fn io_error_message(e: &std::io::Error, resource: &str) -> FactoryError {
    let message = match e.kind() {
        std::io::ErrorKind::IsADirectory => "Is a directory".to_string(),
        std::io::ErrorKind::PermissionDenied => resource.to_string(),
        _ => {
            let text = e.to_string();
            match text.find(" (os error ") {
                Some(i) => text[..i].to_string(),
                None => text,
            }
        }
    };
    FactoryError::io(message)
}

/// An in-memory loader (Rust-only): resources by name, then a delegate.
#[derive(Clone, Default)]
pub struct MapResourceLoader {
    resources: HashMap<String, Arc<[u8]>>,
}

impl MapResourceLoader {
    /// An empty loader; unknown names fall through to the
    /// [`ClasspathResourceLoader`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) a resource.
    pub fn with(mut self, name: &str, bytes: impl AsRef<[u8]>) -> Self {
        self.resources
            .insert(name.to_string(), Arc::from(bytes.as_ref()));
        self
    }
}

impl ResourceLoader for MapResourceLoader {
    fn open_resource(&self, resource: &str) -> Result<Vec<u8>, FactoryError> {
        match self.resources.get(resource) {
            Some(bytes) => Ok(bytes.to_vec()),
            None => ClasspathResourceLoader.open_resource(resource),
        }
    }
}

// ------------------------------------------------- AbstractAnalysisFactory

/// UTF-8 with `CodingErrorAction.REPORT`.
pub(crate) fn decode_utf8(bytes: Vec<u8>) -> Result<String, FactoryError> {
    String::from_utf8(bytes).map_err(|e| {
        let len = e.utf8_error().error_len().unwrap_or(1);
        FactoryError::new(
            JavaException::MalformedInput,
            format!("Input length = {len}"),
        )
    })
}

/// `String.trim()`: strips chars `<= ' '`.
pub(crate) fn java_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c <= ' ')
}

/// `getLines(loader, resource)`: `WordlistLoader.getLines` over the UTF-8
/// resource.
pub fn get_lines(loader: &dyn ResourceLoader, resource: &str) -> Result<Vec<String>, FactoryError> {
    let text = decode_utf8(loader.open_resource(resource)?)?;
    Ok(wordlist_loader::get_lines(text.as_bytes())?)
}

/// `getWordSet(loader, wordFiles, ignoreCase)`: the lines of every
/// comma-separated file; `None` when there is no file name.
pub fn get_word_set(
    loader: &dyn ResourceLoader,
    word_files: &str,
    ignore_case: bool,
) -> Result<Option<CharArraySet>, FactoryError> {
    let files = split_file_names(Some(word_files));
    if files.is_empty() {
        return Ok(None);
    }
    let mut words = CharArraySet::with_capacity(files.len() * 10, ignore_case);
    for file in &files {
        for line in get_lines(loader, java_trim(file))? {
            words.add(&line);
        }
    }
    Ok(Some(words))
}

/// `getSnowballWordSet(loader, wordFiles, ignoreCase)`.
pub fn get_snowball_word_set(
    loader: &dyn ResourceLoader,
    word_files: &str,
    ignore_case: bool,
) -> Result<Option<CharArraySet>, FactoryError> {
    let files = split_file_names(Some(word_files));
    if files.is_empty() {
        return Ok(None);
    }
    let mut words = CharArraySet::with_capacity(files.len() * 10, ignore_case);
    for file in &files {
        let text = decode_utf8(loader.open_resource(java_trim(file))?)?;
        wordlist_loader::get_snowball_word_set_into(text.as_bytes(), &mut words)?;
    }
    Ok(Some(words))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_errors_carry_javas_text_not_paths() {
        use std::io::{Error, ErrorKind};
        let denied = io_error_message(&Error::from(ErrorKind::PermissionDenied), "w.txt");
        assert_eq!(
            (denied.kind, denied.message.as_str()),
            (JavaException::Io, "w.txt")
        );
        let eio = io_error_message(&Error::from_raw_os_error(5), "w.txt");
        assert_eq!(eio.message, "Input/output error");
        let other = io_error_message(&Error::other("odd"), "w.txt");
        assert_eq!(other.message, "odd");
    }

    #[test]
    fn classpath_serves_the_vendored_files() {
        let names: Vec<&str> = ClasspathResourceLoader::resources().collect();
        assert_eq!(names.len(), 42);
        let german = ClasspathResourceLoader
            .open_resource("org/apache/lucene/analysis/snowball/german_stop.txt")
            .unwrap();
        assert!(!german.is_empty());
        for vendored in [
            "org/apache/lucene/analysis/snowball/english_stop.txt",
            "org/apache/lucene/analysis/cjk/stopwords.txt",
        ] {
            assert!(!ClasspathResourceLoader
                .open_resource(vendored)
                .unwrap()
                .is_empty());
        }
        let e = ClasspathResourceLoader
            .open_resource("org/apache/lucene/analysis/compound/hyphenation/hyphenation.dtd")
            .unwrap_err();
        assert_eq!(e.kind, JavaException::Io);
        assert!(e.message.ends_with("hyphenation.dtd"));
    }

    #[test]
    fn filesystem_falls_back_to_its_delegate() {
        let dir = std::env::temp_dir().join(format!("m11p4-loader-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("w.txt"), "#c\n a \nb\n").unwrap();
        let fs = FilesystemResourceLoader::with_classpath(&dir).unwrap();
        assert_eq!(fs.base_directory(), dir.as_path());
        assert_eq!(get_lines(&fs, "w.txt").unwrap(), vec!["a", "b"]);
        assert!(fs
            .open_resource("org/apache/lucene/analysis/ar/stopwords.txt")
            .is_ok());
        assert_eq!(
            fs.open_resource("nope").unwrap_err().kind,
            JavaException::Io
        );
        // A directory is an I/O error, not a fallback.
        assert_eq!(fs.open_resource("sub").unwrap_err().kind, JavaException::Io);
        let e = FilesystemResourceLoader::with_classpath(dir.join("w.txt"))
            .err()
            .unwrap();
        assert!(e.message.ends_with("w.txt is not a directory"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn word_sets() {
        let loader = MapResourceLoader::new()
            .with("a.txt", "Foo\n#x\nbar\n")
            .with("b.txt", "baz | comment\nqux quux\n")
            .with("bad", [0x66, 0xFF, 0x66]);
        let set = get_word_set(&loader, "a.txt, b.txt", true)
            .unwrap()
            .unwrap();
        assert!(set.contains("foo") && set.contains("FOO") && set.contains("bar"));
        assert!(set.contains("baz | comment"));
        assert!(get_word_set(&loader, "", false).is_err());
        let snow = get_snowball_word_set(&loader, "b.txt", false)
            .unwrap()
            .unwrap();
        let mut words: Vec<&str> = snow.iter().collect();
        words.sort();
        assert_eq!(words, vec!["baz", "quux", "qux"]);
        assert!(get_snowball_word_set(&loader, "", false).is_err());
        let e = get_lines(&loader, "bad").unwrap_err();
        assert_eq!(e.kind, JavaException::MalformedInput);
        assert_eq!(e.message, "Input length = 1");
        assert!(get_snowball_word_set(&loader, "bad", false).is_err());
        assert!(loader
            .open_resource("org/apache/lucene/analysis/tr/stopwords.txt")
            .is_ok());
    }
}
