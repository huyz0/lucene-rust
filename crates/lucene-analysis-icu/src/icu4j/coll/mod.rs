//! ICU4J 77.1 collation (`com.ibm.icu.impl.coll`, `RuleBasedCollator`),
//! the sort-key half: ICU's root collation (`coll/ucadata.icu`) and every
//! locale tailoring ICU ships (`coll/*.res`, their `%%CollationBin`),
//! iterated into collation elements and written as sort keys byte for byte.
//!
//! - [`collation`]: `Collation`, the CE and CE32 encodings.
//! - [`crate::icu4j::trie2`]: `Trie2_32`, the CE32 trie.
//! - [`data`]: `CollationData`, `CollationDataReader`, `CollationTailoring`.
//! - [`settings`]: `CollationSettings`, including script reordering.
//! - [`fcd`]: `CollationFCD`, the coarse combining-class sets.
//! - [`iter`]: `CollationIterator`, `UTF16CollationIterator`,
//!   `FCDUTF16CollationIterator` (forward).
//! - [`keys`]: `CollationKeys`, `BOCSU`, the identical level.
//! - [`locale`]: `ULocale` IDs (`LocaleIDParser`).
//! - [`res`]: the data pack, `.res` resource bundles and their fallback.
//! - [`collator`]: `Collator`/`RuleBasedCollator`, `CollationLoader`.

pub mod collation;
pub mod collator;
pub mod data;
pub mod fcd;
pub mod iter;
pub mod keys;
pub mod locale;
pub mod res;
pub mod settings;
