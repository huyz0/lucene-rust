//! `org.apache.lucene.facet.taxonomy.FacetLabel`: a category path, the dim
//! first, then the path components.
//!
//! Java semantics kept: `compareTo` compares components as Java `String`s
//! (UTF-16 code units, so a supplementary character sorts *before* U+FFFF,
//! unlike UTF-8 byte order), `hashCode`/`longHashCode` are Java's
//! (`String.hashCode` per component), the length limit counts UTF-16 units,
//! and a [`FacetLabel::subpath`] shares its parent's components -- so
//! [`FacetLabel::last_component`] of a subpath is, as in Java, the *parent's*
//! last component (it reads `components.length`, not `length`).

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::{Error, Result};

/// `ByteBlockPool.BYTE_BLOCK_SIZE`.
const BYTE_BLOCK_SIZE: usize = 1 << 15;

/// Java's `String.hashCode`, over UTF-16 code units.
pub(crate) fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

/// Java's `String.compareTo`: UTF-16 code unit order. Equal to Rust's byte
/// order except where a supplementary character (a surrogate pair,
/// `0xD800..0xDFFF`) meets a BMP character at or above `0xE000`.
pub(crate) fn java_compare(a: &str, b: &str) -> Ordering {
    // Fast path: Rust's byte order is code point order, which only differs
    // from UTF-16 order when a 4-byte UTF-8 sequence is involved.
    if a.is_ascii() || b.is_ascii() {
        return a.cmp(b);
    }
    a.encode_utf16().cmp(b.encode_utf16())
}

/// `String.length()`: the UTF-16 length.
pub(crate) fn utf16_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    }
}

/// `Arrays.toString(String[])`.
pub(crate) fn java_array_to_string<S: AsRef<str>>(items: &[S]) -> String {
    let mut out = String::from("[");
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(s.as_ref());
    }
    out.push(']');
    out
}

/// A component's hashes, computed once as Java's `String` caches its
/// `hashCode`: the Java hash ([`FacetLabel::hash_code`],
/// [`FacetLabel::long_hash_code`]) and the word [`Hash`] folds.
#[derive(Clone, Copy)]
struct ComponentHash {
    java: i32,
    word: u64,
}

const MIX: u64 = 0x51_7c_c1_b7_27_22_0a_95;

fn mix(h: u64, w: u64) -> u64 {
    (h.rotate_left(5) ^ w).wrapping_mul(MIX)
}

impl ComponentHash {
    /// The bytes folded eight at a time after their length (so `["ab",
    /// "c"]` and `["a", "bc"]` differ).
    fn of(s: &str) -> Self {
        let bytes = s.as_bytes();
        let mut h = mix(0, bytes.len() as u64);
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut w = [0u8; 8];
            w.copy_from_slice(chunk);
            h = mix(h, u64::from_le_bytes(w));
        }
        let tail = chunks.remainder();
        if !tail.is_empty() {
            let mut w = [0u8; 8];
            w[..tail.len()].copy_from_slice(tail);
            h = mix(h, u64::from_le_bytes(w));
        }
        ComponentHash {
            java: java_string_hash(s),
            word: h,
        }
    }
}

/// `FacetLabel`.
#[derive(Clone)]
pub struct FacetLabel {
    /// `components`: shared with every [`FacetLabel::subpath`] taken of it.
    components: Arc<[String]>,
    /// Each component's hashes, shared the same way.
    hashes: Arc<[ComponentHash]>,
    /// `length`: how many of `components` this label uses.
    length: usize,
}

impl FacetLabel {
    /// `MAX_CATEGORY_PATH_LENGTH`: `(BYTE_BLOCK_SIZE - 2) / 4`, the longest
    /// path in UTF-16 units, separators included.
    pub const MAX_CATEGORY_PATH_LENGTH: usize = (BYTE_BLOCK_SIZE - 2) / 4;

    /// `new FacetLabel(String... components)`.
    pub fn new<S: Into<String>>(components: impl IntoIterator<Item = S>) -> Result<Self> {
        let components: Arc<[String]> = components.into_iter().map(Into::into).collect();
        check_components(&components)?;
        let length = components.len();
        let hashes = components.iter().map(|c| ComponentHash::of(c)).collect();
        Ok(FacetLabel {
            components,
            hashes,
            length,
        })
    }

    /// `new FacetLabel(String dim, String[] path)`.
    pub fn from_dim_path<S: AsRef<str>>(dim: &str, path: &[S]) -> Result<Self> {
        Self::new(std::iter::once(dim).chain(path.iter().map(AsRef::as_ref)))
    }

    /// `new FacetLabel()`: the root, with no components.
    pub fn root() -> Self {
        FacetLabel {
            components: Arc::from(Vec::<String>::new()),
            hashes: Arc::from(Vec::<ComponentHash>::new()),
            length: 0,
        }
    }

    /// `components`: the whole shared array, which for a subpath is longer
    /// than [`FacetLabel::len`].
    pub fn components(&self) -> &[String] {
        &self.components
    }

    /// The components this label uses (`components[0..length]`).
    pub fn path(&self) -> &[String] {
        &self.components[..self.length]
    }

    /// `length`.
    pub fn len(&self) -> usize {
        self.length
    }

    /// `length == 0`: the root.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// `compareTo`: component by component as Java strings, then the shorter
    /// first.
    pub fn compare_to(&self, other: &FacetLabel) -> Ordering {
        for (a, b) in self.path().iter().zip(other.path()) {
            match java_compare(a, b) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }
        self.length.cmp(&other.length)
    }

    /// `hashCode`.
    pub fn hash_code(&self) -> i32 {
        if self.length == 0 {
            return 0;
        }
        let mut hash = i32::try_from(self.length).unwrap_or(i32::MAX);
        for c in &self.hashes[..self.length] {
            hash = hash.wrapping_mul(31).wrapping_add(c.java);
        }
        hash
    }

    /// `longHashCode`.
    pub fn long_hash_code(&self) -> i64 {
        if self.length == 0 {
            return 0;
        }
        let mut hash = i64::try_from(self.length).unwrap_or(i64::MAX);
        for c in &self.hashes[..self.length] {
            hash = hash.wrapping_mul(65599).wrapping_add(i64::from(c.java));
        }
        hash
    }

    /// `subpath(length)`: the first `length` components, sharing this
    /// label's array; this label itself when `length` is not shorter.
    pub fn subpath(&self, length: usize) -> FacetLabel {
        if length >= self.length {
            self.clone()
        } else {
            FacetLabel {
                components: Arc::clone(&self.components),
                hashes: Arc::clone(&self.hashes),
                length,
            }
        }
    }

    /// `lastComponent()`: the last of `components` -- the whole shared
    /// array's, as in Java, so a subpath answers its parent's.
    pub fn last_component(&self) -> Result<&str> {
        self.components
            .last()
            .map(String::as_str)
            .ok_or_else(|| Error::Unsupported("components is empty".to_string()))
    }

    /// `FacetsConfig.pathToString(components, length)`: the label's term in
    /// the taxonomy index. A label's components are never empty
    /// ([`FacetLabel::new`] refuses them), so the encoding cannot fail.
    pub(crate) fn to_term(&self) -> String {
        let mut out = String::new();
        for (i, s) in self.path().iter().enumerate() {
            if i > 0 {
                out.push(crate::facets::DELIM_CHAR);
            }
            for ch in s.chars() {
                if ch == crate::facets::DELIM_CHAR || ch == crate::facets::ESCAPE_CHAR {
                    out.push(crate::facets::ESCAPE_CHAR);
                }
                out.push(ch);
            }
        }
        out
    }

    /// `new FacetLabel(FacetsConfig.stringToPath(term.utf8ToString()))`.
    pub(crate) fn from_term(term: &[u8]) -> Result<Self> {
        Self::new(crate::facets::string_to_path(&String::from_utf8_lossy(
            term,
        )))
    }
}

/// `checkComponents()`.
fn check_components(components: &[String]) -> Result<()> {
    let mut len: usize = 0;
    for c in components {
        if c.is_empty() {
            return Err(Error::IllegalArgument(format!(
                "empty or null components not allowed: {}",
                java_array_to_string(components)
            )));
        }
        len = len.saturating_add(utf16_len(c));
    }
    // The separators.
    len = len.saturating_add(components.len().saturating_sub(1));
    if len > FacetLabel::MAX_CATEGORY_PATH_LENGTH {
        let shown: Vec<u16> = java_array_to_string(components)
            .encode_utf16()
            .take(30)
            .collect();
        return Err(Error::IllegalArgument(format!(
            "category path exceeds maximum allowed path length: max={} len={} path={}...",
            FacetLabel::MAX_CATEGORY_PATH_LENGTH,
            len,
            String::from_utf16_lossy(&shown)
        )));
    }
    Ok(())
}

impl PartialEq for FacetLabel {
    /// `equals`: same length, equal components. A label and its clone (the
    /// same shared array) are equal without reading the strings, as Java's
    /// `HashMap` finds the very key it was given by identity.
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }
        if Arc::ptr_eq(&self.components, &other.components) {
            return true;
        }
        let same_words = self.hashes[..self.length]
            .iter()
            .zip(&other.hashes[..other.length])
            .all(|(a, b)| a.word == b.word);
        same_words && self.path() == other.path()
    }
}

impl Eq for FacetLabel {}

impl Hash for FacetLabel {
    /// One word per label, folded from the components' cached hashes: the
    /// writer and reader caches hash a label per lookup, and hashing the
    /// strings themselves each time was most of a cached `addCategory`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut h = self.length as u64;
        for c in &self.hashes[..self.length] {
            h = mix(h, c.word);
        }
        state.write_u64(h);
    }
}

impl PartialOrd for FacetLabel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FacetLabel {
    /// [`FacetLabel::compare_to`].
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare_to(other)
    }
}

impl fmt::Display for FacetLabel {
    /// `toString()`: `FacetLabel: [a, b]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FacetLabel: {}", java_array_to_string(self.path()))
    }
}

impl fmt::Debug for FacetLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests;
