//! `org.apache.lucene.luke.models`: what Luke's index inspector reports,
//! without its Swing front end. Each Java model is a struct over an opened
//! [`DirectoryReader`] (or a directory, for [`tools::IndexTools`]):
//!
//! | Java | Rust |
//! |------|------|
//! | `OverviewImpl` | [`overview::Overview`] |
//! | `DocumentsImpl` | [`documents::Documents`] |
//! | `CommitsImpl` | [`commits::Commits`] |
//! | `SearchImpl` | [`search::Search`] |
//! | `AnalysisImpl` | [`analysis::Analysis`] |
//! | `IndexToolsImpl` | [`tools::IndexTools`] |
//! | `util.twentynewsgroups` | [`newsgroups`] |
//!
//! [`cli`] is the `Luke` tool: the models through a command line, the twin of
//! `fixtures/src/GenToolsLuke.java`.
//!
//! Differs, for every model: Luke opens every index under a path
//! (`IndexUtils.openIndex` walks it and wraps several in a `MultiReader`) in a
//! directory implementation the user names; here a model reads the one index
//! in a directory, through [`lucene_store::directory::FsDirectory`].
//!
//! [`DirectoryReader`]: lucene_search::directory_reader::DirectoryReader

pub mod analysis;
pub mod cli;
pub mod commits;
pub mod documents;
pub mod newsgroups;
pub mod overview;
pub mod search;
pub mod tools;

/// `LukeException`, or the Java exception a model lets through: its simple
/// class name and its message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LukeError {
    /// `getClass().getSimpleName()`.
    pub class: &'static str,
    /// `getMessage()`; `None` for Java's `null`.
    pub message: Option<String>,
}

impl LukeError {
    /// `new LukeException(message)`.
    pub fn luke(message: impl Into<String>) -> Self {
        LukeError {
            class: "LukeException",
            message: Some(message.into()),
        }
    }

    /// `new LukeException(cause)`: the cause's `toString()` is the message.
    pub fn luke_caused_by(class: &str, message: &str) -> Self {
        Self::luke(format!("{class}: {message}"))
    }

    /// `new IllegalArgumentException(message)`.
    pub fn illegal_argument(message: impl Into<String>) -> Self {
        LukeError {
            class: "IllegalArgumentException",
            message: Some(message.into()),
        }
    }

    /// `new IllegalStateException()`.
    pub fn illegal_state() -> Self {
        LukeError {
            class: "IllegalStateException",
            message: None,
        }
    }
}

impl std::fmt::Display for LukeError {
    /// `getSimpleName() + ": " + getMessage()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {}",
            self.class,
            self.message.as_deref().unwrap_or("null")
        )
    }
}

impl std::error::Error for LukeError {}

/// A model's result.
pub type Result<T> = std::result::Result<T, LukeError>;

/// `BytesRef.toString()`: the bytes in lower-case hex, space separated, in
/// brackets.
pub fn bytes_ref_string(bytes: &[u8]) -> String {
    let hex: Vec<String> = bytes.iter().map(|b| format!("{b:x}")).collect();
    format!("[{}]", hex.join(" "))
}

/// `BytesRefUtils.decode`: the bytes as UTF-8, or [`bytes_ref_string`] when
/// they are not UTF-8.
pub fn decode(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => bytes_ref_string(bytes),
    }
}

/// `AbstractMap.toString()` of the map `DataInput.readMapOfStrings` returns
/// for `entries` (in their order on disk): sorted by key up to ten entries (a
/// `TreeMap`), in `HashMap` order above that.
pub fn read_map_string(entries: &[(String, String)]) -> String {
    let order: Vec<usize> = if entries.len() > 10 {
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
        lucene_search::mlt::java_hash_order(&keys)
    } else {
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by(|&a, &b| entries[a].0.cmp(&entries[b].0));
        order
    };
    map_string(order.iter().map(|&i| (&entries[i].0, &entries[i].1)))
}

/// `AbstractMap.toString()` of entries in iteration order.
pub fn map_string<K: std::fmt::Display, V: std::fmt::Display>(
    entries: impl IntoIterator<Item = (K, V)>,
) -> String {
    let parts: Vec<String> = entries
        .into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    format!("{{{}}}", parts.join(", "))
}

/// `AbstractCollection.toString()`.
pub fn list_string<T: std::fmt::Display>(items: impl IntoIterator<Item = T>) -> String {
    let parts: Vec<String> = items.into_iter().map(|i| i.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// `CommitsImpl.toDisplaySize`: bytes, kilobytes or megabytes, rounded down.
pub fn display_size(size: u64) -> String {
    if size < 1024 {
        format!("{size} B")
    } else if size < 1_048_576 {
        format!("{} KB", size / 1024)
    } else {
        format!("{} MB", size / 1_048_576)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_print_as_java_names_them() {
        assert_eq!(LukeError::luke("x").to_string(), "LukeException: x");
        assert_eq!(
            LukeError::luke_caused_by("java.lang.IllegalStateException", "y").to_string(),
            "LukeException: java.lang.IllegalStateException: y"
        );
        assert_eq!(
            LukeError::illegal_argument("z").to_string(),
            "IllegalArgumentException: z"
        );
        assert_eq!(
            LukeError::illegal_state().to_string(),
            "IllegalStateException: null"
        );
    }

    #[test]
    fn bytes_print_as_bytes_ref_does() {
        assert_eq!(bytes_ref_string(&[0x80, 0, 0x6f]), "[80 0 6f]");
        assert_eq!(bytes_ref_string(&[]), "[]");
        assert_eq!(decode(b"abc"), "abc");
        assert_eq!(decode(&[0xff, 0x01]), "[ff 1]");
    }

    #[test]
    fn small_maps_are_sorted_and_large_ones_hashed() {
        let entries: Vec<(String, String)> = [("b", "2"), ("a", "1")]
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        assert_eq!(read_map_string(&entries), "{a=1, b=2}");
        assert_eq!(read_map_string(&[]), "{}");
        let many: Vec<(String, String)> =
            (0..12).map(|i| (format!("k{i}"), i.to_string())).collect();
        let s = read_map_string(&many);
        // `java.util.HashMap` buckets "k0".."k9" by hash, "k10"/"k11" after
        assert!(s.starts_with("{k0=0, k1=1, "), "{s}");
        assert_eq!(s.matches('=').count(), 12);
    }

    #[test]
    fn sizes_round_down_to_their_unit() {
        assert_eq!(display_size(1023), "1023 B");
        assert_eq!(display_size(1024), "1 KB");
        assert_eq!(display_size(1_048_575), "1023 KB");
        assert_eq!(display_size(3 * 1_048_576 + 5), "3 MB");
        assert_eq!(list_string(["a", "b"]), "[a, b]");
        assert_eq!(map_string([("a", 1)]), "{a=1}");
    }
}
