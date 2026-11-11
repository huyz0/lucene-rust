//! The plugin's phrase query over a `pattern_text` field, as node `47` of the
//! query tree ([`crate::jvm_queries`]).
//!
//! The phrase's terms are a filter-only conjunction, and each candidate is
//! verified against its message as [`lucene_pattern_text::PatternPhraseSource`]
//! verifies it: once per template where the template settles it, otherwise by
//! rebuilding the message from the template and the stored variables and
//! running the field's index analyzer over it. Every hit scores 0, as the
//! plugin's Java `PatternPhraseQuery` and a `match_only_text` phrase score.
//!
//! The node, after its kind byte:
//!
//! | field | layout |
//! |---|---|
//! | `field` | string |
//! | analyzer | bytes: the index analyzer's spec ([`crate::jvm_analysis`]) |
//! | `template_parts` | `u8`: 1 to decide from a template's static text, 0 not to |
//! | positions | `i32` count, at least 1, then per position `position: i32` (non-decreasing), `n: i32` (at least 1) and `n` terms as bytes |

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use lucene_analysis::TokenStream;
use lucene_pattern_text::{pattern_phrase_query, PatternPhraseSource, PhraseTokens, Tokenizer};
use lucene_search::query::Clause;

use crate::error::FfiStatus;
use crate::jvm_nodes::invalid;
use crate::jvm_reader::Cursor;

pub(crate) const NODE_PATTERN_PHRASE: u8 = 47;

/// At most this many terms in one phrase (the plugin's `MAX_PATTERN_TERMS`
/// keeps a longer one in Java).
const MAX_TERMS: usize = 1024;

fn string(c: &mut Cursor<'_>) -> std::result::Result<String, FfiStatus> {
    Ok(std::str::from_utf8(c.bytes()?)
        .map_err(|_| FfiStatus::InvalidUtf8)?
        .to_string())
}

pub(crate) fn decode(c: &mut Cursor<'_>) -> std::result::Result<Clause, FfiStatus> {
    let field = string(c)?;
    let spec = c.bytes()?;
    let template_parts = match c.u8()? {
        0 => false,
        1 => true,
        f => {
            return Err(invalid(format!(
                "query tree: pattern phrase template flag {f} (expected 0..=1)"
            )))
        }
    };
    let count = c.len()?;
    if count == 0 {
        return Err(invalid("query tree: pattern phrase without terms"));
    }
    let mut tokens: Vec<(i32, String)> = Vec::new();
    let mut last = i32::MIN;
    for _ in 0..count {
        let pos = c.i32()?;
        if pos < last {
            return Err(invalid("query tree: pattern phrase positions out of order"));
        }
        last = pos;
        let n = c.len()?;
        if n == 0 {
            return Err(invalid("query tree: pattern phrase position without terms"));
        }
        if tokens.len().saturating_add(n) > MAX_TERMS {
            return Err(invalid(format!(
                "query tree: pattern phrase with more than {MAX_TERMS} terms"
            )));
        }
        for _ in 0..n {
            tokens.push((pos, string(c)?));
        }
    }
    let analyzer = crate::jvm_analysis::analyzer(spec)?;
    let mut hasher = DefaultHasher::new();
    spec.hash(&mut hasher);
    let key = format!("{:016x}", hasher.finish());
    let name = field.clone();
    // A message the analyzer refuses keeps the tokens read before the
    // failure, so its document can only miss, never match wrongly.
    let tokenizer: Arc<Tokenizer> = Arc::new(move |text: &str, out: &mut Vec<(i32, String)>| {
        let Ok(mut ts) = analyzer.analyzer().token_stream(&name, text) else {
            return;
        };
        let mut pos = -1i32;
        if ts.reset().is_err() {
            return;
        }
        while let Ok(true) = ts.increment_token() {
            let a = ts.attributes();
            pos = pos.saturating_add(a.position_increment());
            out.push((pos, a.term().to_string()));
        }
        let _ = ts.end();
        let _ = ts.close();
    });
    let phrase = PhraseTokens::new(tokens);
    let mut verifier = PatternPhraseSource::new(&field, phrase.clone(), tokenizer, key);
    if !template_parts {
        verifier = verifier.whole_messages_only();
    }
    Ok(pattern_phrase_query(&field, phrase, verifier))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_analysis::tests::spec;
    use crate::jvm_nodes::tests::{fails, open, search, Blob};
    use crate::jvm_reader::ffi_close_jvm_reader;

    fn standard() -> Vec<u8> {
        spec(&[], ("standard", &[]), &[("lowercase", &[])], &[])
    }

    fn node(template_parts: u8, phrase: &[(i32, &[&str])]) -> Blob {
        let mut b = Blob::tree()
            .u8(NODE_PATTERN_PHRASE)
            .str("message")
            .bytes(&standard())
            .u8(template_parts)
            .i32(phrase.len() as i32);
        for (p, terms) in phrase {
            b = b.i32(*p).i32(terms.len() as i32);
            for t in *terms {
                b = b.str(t);
            }
        }
        b
    }

    fn decoded(blob: &Blob) -> std::result::Result<Clause, FfiStatus> {
        let mut c = Cursor::new(&blob.0[2..]);
        decode(&mut c)
    }

    #[test]
    fn matches_the_messages_that_hold_the_phrase() {
        use lucene_index::document::{Document, Store, StringField};
        use lucene_index::index_writer::IndexWriter;
        use lucene_store::FsDirectory;
        use lucene_util::test_support::TempDir;

        let messages = [
            "Failed password for root from 10.0.0.1 port 22 ssh2",
            "Failed password for invalid user admin1 from 10.0.0.2",
            "connection 3 closed by peer",
            "connection 4 closed by peer",
            "instruction cache parity error corrected",
            "password Failed",
        ];
        let tmp = TempDir::new("pattern_phrase_node");
        let dir = FsDirectory::open(tmp.path());
        let version = lucene_index::segment_info::LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        for (i, m) in messages.iter().enumerate() {
            let mut d = Document::new();
            d.add(StringField::new("id", i.to_string(), Store::Yes));
            lucene_pattern_text::fields(&mut d, "message", m).unwrap();
            w.add_fields_documents(&[d]).unwrap();
        }
        w.commit().unwrap();
        let h = open(&tmp);
        let docs = |blob: &Blob| {
            let (hits, total) = search(h, blob);
            assert!(hits.iter().all(|&(_, s)| s == 0.0), "phrases score 0");
            let mut d: Vec<i32> = hits.into_iter().map(|(d, _)| d).collect();
            d.sort_unstable();
            assert_eq!(total, d.len() as i64);
            d
        };
        for parts in [0, 1] {
            let failed: &[(i32, &[&str])] = &[(0, &["failed"]), (1, &["password"])];
            assert_eq!(docs(&node(parts, failed)), [0, 1]);
            let var: &[(i32, &[&str])] = &[(0, &["connection"]), (1, &["3"]), (2, &["closed"])];
            assert_eq!(docs(&node(parts, var)), [2]);
            // stacked terms at one position match either
            let either: &[(i32, &[&str])] = &[(0, &["3", "4"]), (1, &["closed"])];
            assert_eq!(docs(&node(parts, either)), [2, 3]);
            let none: &[(i32, &[&str])] = &[(0, &["parity"]), (1, &["corrected"])];
            assert!(docs(&node(parts, none)).is_empty());
            let one: &[(i32, &[&str])] = &[(5, &["parity"])];
            assert_eq!(docs(&node(parts, one)), [4]);
        }
        let (_, msg) = fails(h, &node(2, &[(0, &["a"])]));
        assert!(msg.contains("template flag"), "{msg}");
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn refuses_bad_nodes() {
        assert!(decoded(&node(1, &[(0, &["a"]), (1, &["b"])])).is_ok());
        for bad in [
            node(1, &[]),
            node(1, &[(0, &[])]),
            node(1, &[(1, &["a"]), (0, &["b"])]),
        ] {
            assert!(decoded(&bad).is_err());
        }
        let many: Vec<&str> = vec!["t"; MAX_TERMS + 1];
        assert!(decoded(&node(1, &[(0, &many)])).is_err());
    }
}
