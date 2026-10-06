//! `org.apache.lucene.analysis.synonym.WordnetSynonymParser`: WordNet's
//! prolog `wn_s.pl` lines, `s(100001740,1,'entity',n,1,11).`.
//!
//! Consecutive lines with the same synset id (chars 2..11) form one synset;
//! the quoted word (between the first and last `'`, `''` read as `'`) is
//! analyzed into a phrase. With `expand`, every phrase of a synset maps to
//! every other one, keeping the original; without, every phrase maps to the
//! first, dropping it.
//!
//! Differs: [`WordnetSynonymParser::parse`] takes a `&str`; a line Java's
//! `substring` fails on (shorter than 11 chars, or no quote) is
//! [`SynonymParseError::Malformed`] where Java throws an unwrapped
//! `StringIndexOutOfBoundsException`.

use crate::analyzer::Analyzer;

use super::synonym_map::{java_lines, SynonymMap, SynonymParseError, SynonymParserBase};

/// `WordnetSynonymParser`.
pub struct WordnetSynonymParser<'a> {
    base: SynonymParserBase<'a>,
    expand: bool,
}

/// `s.substring(begin, end)` over UTF-16 units, `None` where Java throws.
fn utf16_substring(units: &[u16], begin: i64, end: i64) -> Option<String> {
    if begin < 0 || end > units.len() as i64 || begin > end {
        return None;
    }
    Some(String::from_utf16_lossy(
        &units[begin as usize..end as usize],
    ))
}

impl<'a> WordnetSynonymParser<'a> {
    /// `new WordnetSynonymParser(dedup, expand, analyzer)`.
    pub fn new(dedup: bool, expand: bool, analyzer: &'a Analyzer) -> Self {
        WordnetSynonymParser {
            base: SynonymParserBase::new(dedup, analyzer),
            expand,
        }
    }

    /// `parse(Reader)`.
    pub fn parse(&mut self, rules: &str) -> Result<(), SynonymParseError> {
        let mut last_synset_id = String::new();
        let mut synset: Vec<String> = Vec::new();
        for (i, line) in java_lines(rules).into_iter().enumerate() {
            let line_no = i + 1;
            let units: Vec<u16> = line.encode_utf16().collect();
            let synset_id = utf16_substring(&units, 2, 11)
                .ok_or(SynonymParseError::Malformed { line: line_no })?;
            if synset_id != last_synset_id {
                self.add_internal(&synset)
                    .map_err(|cause| SynonymParseError::InvalidRule {
                        line: line_no,
                        cause,
                    })?;
                synset.clear();
            }
            // Java: parseSynonym
            let start = units
                .iter()
                .position(|&u| u == u16::from(b'\''))
                .map_or(0, |p| p + 1);
            let end = units
                .iter()
                .rposition(|&u| u == u16::from(b'\''))
                .map_or(-1, |p| p as i64);
            let text = utf16_substring(&units, start as i64, end)
                .ok_or(SynonymParseError::Malformed { line: line_no })?
                .replace("''", "'");
            let phrase =
                self.base
                    .analyze(&text)
                    .map_err(|cause| SynonymParseError::InvalidRule {
                        line: line_no,
                        cause,
                    })?;
            synset.push(phrase);
            last_synset_id = synset_id;
        }
        let lines = java_lines(rules).len();
        self.add_internal(&synset)
            .map_err(|cause| SynonymParseError::InvalidRule { line: lines, cause })
    }

    /// `build()` (see [`SynonymMapBuilder::build`](super::SynonymMapBuilder::build)).
    pub fn build(self) -> Result<SynonymMap, crate::AnalysisError> {
        self.base.builder.build()
    }

    // Java: WordnetSynonymParser.addInternal
    fn add_internal(&mut self, synset: &[String]) -> Result<(), crate::AnalysisError> {
        if synset.len() <= 1 {
            return Ok(());
        }
        let b = &mut self.base.builder;
        if self.expand {
            for (i, a) in synset.iter().enumerate() {
                for (j, o) in synset.iter().enumerate() {
                    if i != j {
                        b.add(a, o, true)?;
                    }
                }
            }
        } else {
            for a in synset {
                b.add(a, &synset[0], false)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dump(m: &SynonymMap) -> Vec<(String, bool, Vec<String>)> {
        m.entries()
            .into_iter()
            .map(|(k, e)| {
                (
                    k,
                    e.keep_orig,
                    e.ords.iter().map(|&o| m.word(o).to_string()).collect(),
                )
            })
            .collect()
    }

    const RULES: &str = "s(100000001,1,'woods',n,1,0).\n\
                         s(100000001,2,'wood',n,1,0).\n\
                         s(100000001,3,'forest',n,1,0).\n\
                         s(100000002,1,'king''s evil',n,1,0).\n\
                         s(100000002,2,'scrofula',n,1,0).\n\
                         s(100000003,1,'lone',n,1,0).";

    #[test]
    fn synsets_expand_or_collapse() {
        let ws = Analyzer::new(crate::core_analysis::WhitespaceAnalyzer::default());
        let mut p = WordnetSynonymParser::new(true, true, &ws);
        p.parse(RULES).unwrap();
        let d = dump(&p.build().unwrap());
        assert_eq!(d.len(), 5);
        assert_eq!(
            d[1],
            (
                "king's\0evil".to_string(),
                true,
                vec!["scrofula".to_string()]
            )
        );
        let mut p = WordnetSynonymParser::new(true, false, &ws);
        p.parse(RULES).unwrap();
        let d = dump(&p.build().unwrap());
        assert!(d.iter().all(|(_, keep, _)| !keep));
        assert_eq!(d.len(), 5);
    }

    #[test]
    fn malformed_and_invalid_lines() {
        let ws = Analyzer::new(crate::core_analysis::WhitespaceAnalyzer::default());
        let mut p = WordnetSynonymParser::new(true, true, &ws);
        assert_eq!(
            p.parse("s(1"),
            Err(SynonymParseError::Malformed { line: 1 })
        );
        let mut p = WordnetSynonymParser::new(true, true, &ws);
        assert_eq!(
            p.parse("s(100000001,1,woods,n,1,0)."),
            Err(SynonymParseError::Malformed { line: 1 })
        );
        let mut p = WordnetSynonymParser::new(true, true, &ws);
        assert!(matches!(
            p.parse("s(100000001,1,' ',n,1,0)."),
            Err(SynonymParseError::InvalidRule { line: 1, .. })
        ));
    }
}
