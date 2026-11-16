//! Text shared with its owner instead of copied. Rust-only: Java's `String`
//! is immutable and shared by reference already.

use std::sync::Arc;

/// Holds either an `Arc<str>` or an `Arc<String>`.
///
/// The second exists so a caller that already built the text in a `String`
/// (a parser reading a 200 MiB field, say) hands it over as it is:
/// `Arc::from(String)` for an `Arc<str>` allocates the text again and copies
/// it, which doubles the peak for as long as both live, while
/// `Arc::new(String)` moves three words. Compares by content, so two handles
/// to equal text are equal whichever form they hold.
#[derive(Debug, Clone)]
pub enum SharedText {
    Str(Arc<str>),
    String(Arc<String>),
}

impl SharedText {
    /// The text.
    pub fn as_str(&self) -> &str {
        match self {
            SharedText::Str(s) => s,
            SharedText::String(s) => s,
        }
    }

    /// Whether both handles share one allocation.
    pub fn ptr_eq(&self, other: &SharedText) -> bool {
        match (self, other) {
            (SharedText::Str(a), SharedText::Str(b)) => Arc::ptr_eq(a, b),
            (SharedText::String(a), SharedText::String(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl std::ops::Deref for SharedText {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for SharedText {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for SharedText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq for SharedText {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for SharedText {}

impl From<Arc<str>> for SharedText {
    fn from(s: Arc<str>) -> Self {
        SharedText::Str(s)
    }
}

impl From<Arc<String>> for SharedText {
    fn from(s: Arc<String>) -> Self {
        SharedText::String(s)
    }
}

/// Takes the `String` as it is; no copy of the text.
impl From<String> for SharedText {
    fn from(s: String) -> Self {
        SharedText::String(Arc::new(s))
    }
}

impl From<&str> for SharedText {
    fn from(s: &str) -> Self {
        SharedText::Str(Arc::from(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_forms_read_as_the_same_text_and_compare_by_content() {
        let a: SharedText = Arc::<str>::from("héllo").into();
        let b: SharedText = Arc::new(String::from("héllo")).into();
        assert_eq!(a.as_str(), "héllo");
        assert_eq!(&*b, "héllo");
        assert_eq!(a.as_ref(), b.as_ref());
        assert_eq!(a, b);
        assert_ne!(a, SharedText::from("other"));
        assert_eq!(format!("{a}"), "héllo");
        assert_eq!(a.len(), "héllo".len());
    }

    #[test]
    fn a_string_is_taken_without_copying_its_bytes() {
        let text = "x".repeat(1 << 20);
        let ptr = text.as_ptr();
        let shared = SharedText::from(text);
        assert_eq!(shared.as_ptr(), ptr, "the String's buffer is the text");
    }

    #[test]
    fn ptr_eq_means_one_allocation_in_the_same_form() {
        let arc = Arc::new(String::from("t"));
        let a = SharedText::from(Arc::clone(&arc));
        let b = SharedText::from(Arc::clone(&arc));
        assert!(a.ptr_eq(&b));
        assert!(!a.ptr_eq(&SharedText::from(String::from("t"))));
        let s: Arc<str> = Arc::from("t");
        let c = SharedText::from(Arc::clone(&s));
        assert!(c.ptr_eq(&SharedText::from(s)));
        assert!(!a.ptr_eq(&c), "different forms never share");
    }
}
