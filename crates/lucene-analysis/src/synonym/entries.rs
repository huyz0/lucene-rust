//! A [`SynonymMap`]'s entries written out as they are, for a map that no
//! rule file describes: one line per input, `keepOrig input output...`,
//! tab-separated, every phrase the hex of its UTF-8 (words joined by
//! [`WORD_SEPARATOR`](super::WORD_SEPARATOR)). Rust-only: Java has no such
//! format. The OpenSearch plugin writes it from a built map (the synonym
//! filters' `synonyms`), so a filter built from it matches that map's,
//! `includeOrig` and output order included.

use super::{SynonymMap, SynonymMapBuilder};
use crate::AnalysisError;

/// The `format` naming this parser to the synonym filter factories.
pub const ENTRIES_FORMAT: &str = "lucene_rust.synonym_entries";

fn bad(line: usize, why: &str) -> AnalysisError {
    AnalysisError::IllegalArgument(format!("synonym entries line {line}: {why}"))
}

fn phrase(hex: &str, line: usize) -> Result<String, AnalysisError> {
    if !hex.len().is_multiple_of(2) {
        return Err(bad(line, "odd hex length"));
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let b = hex
            .get(i..i + 2)
            .and_then(|h| u8::from_str_radix(h, 16).ok())
            .ok_or_else(|| bad(line, "not hex"))?;
        bytes.push(b);
    }
    String::from_utf8(bytes).map_err(|_| bad(line, "not UTF-8"))
}

/// Builds the map `text` lists (no de-duplication: the outputs as listed).
pub fn parse_entries(text: &str) -> Result<SynonymMap, AnalysisError> {
    let mut builder = SynonymMapBuilder::new(false);
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        let keep = match fields.next() {
            Some("0") => false,
            Some("1") => true,
            _ => return Err(bad(i + 1, "keepOrig is not 0 or 1")),
        };
        let input = phrase(fields.next().unwrap_or(""), i + 1)?;
        let mut any = false;
        for out in fields {
            builder.add(&input, &phrase(out, i + 1)?, keep)?;
            any = true;
        }
        if !any {
            return Err(bad(i + 1, "no output"));
        }
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> String {
        s.bytes().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn entries_round_trip_into_a_map() {
        let text = format!(
            "1\t{}\t{}\t{}\n0\t{}\t{}\n",
            hex("ny"),
            hex("new\0york"),
            hex("nyc"),
            hex("a,b"),
            hex("c")
        );
        let map = parse_entries(&text).unwrap();
        assert!(!map.is_empty());
        assert!(parse_entries("2\t61\t62").is_err());
        assert!(parse_entries("0\t61").is_err());
        assert!(parse_entries("0\t6\t62").is_err());
        assert!(parse_entries("").unwrap().is_empty());
    }
}
