//! `TokenStreamToDot`: a token stream's graph as graphviz `dot` text.
//!
//! Java writes to a `PrintWriter` and lets subclasses override the header,
//! trailer, node and arc writers; the port returns the text, with Java's
//! output (`println` is `\n`).

use std::fmt::Write as _;

use lucene_analysis::{AnalysisError, TokenStream};

const FONT_NAME: &str = "Helvetica";

/// `org.apache.lucene.tests.analysis.TokenStreamToDot`.
pub struct TokenStreamToDot<'a> {
    input_text: Option<&'a str>,
    out: String,
}

impl<'a> TokenStreamToDot<'a> {
    /// `TokenStreamToDot(inputText, in, out)`: with `input_text`, each arc
    /// whose term differs from its surface form shows the form too; without
    /// it, the offsets.
    pub fn new(input_text: Option<&'a str>) -> Self {
        TokenStreamToDot {
            input_text,
            out: String::new(),
        }
    }

    /// `toDot()`: consumes `ts` (reset through end) and returns the graph.
    ///
    /// # Errors
    /// What the stream returns.
    pub fn to_dot(mut self, ts: &mut dyn TokenStream) -> Result<String, AnalysisError> {
        ts.reset()?;
        self.write_header();
        let mut pos = -1;
        let mut last_end_pos = -1;
        while ts.increment_token()? {
            let atts = ts.attributes();
            let is_first = pos == -1;
            let mut pos_inc = atts.position_increment();
            if is_first && pos_inc == 0 {
                eprintln!("WARNING: first posInc was 0; correcting to 1");
                pos_inc = 1;
            }
            if pos_inc > 0 {
                // New node:
                pos += pos_inc;
                self.write_node(pos, Some(&pos.to_string()));
            }
            if pos_inc > 1 {
                // Gap!
                self.write_arc(last_end_pos, pos, None, Some("dotted"));
            }
            if is_first {
                self.write_node(-1, None);
                self.write_arc(-1, pos, None, None);
            }
            let term = atts.term();
            let mut arc_label = term.to_string();
            let (start, end) = (atts.start_offset(), atts.end_offset());
            match self.input_text {
                Some(text) => {
                    let units: Vec<u16> = text.encode_utf16().collect();
                    let fragment = units
                        .get(start as usize..end as usize)
                        .map(String::from_utf16_lossy)
                        .ok_or_else(|| {
                            AnalysisError::IllegalArgument(format!(
                                "begin {start}, end {end}, length {}",
                                units.len()
                            ))
                        })?;
                    if fragment != term {
                        arc_label.push_str(" / ");
                        arc_label.push_str(&fragment);
                    }
                }
                None => {
                    let _ = write!(arc_label, " / {start}-{end}");
                }
            }
            let to = pos + atts.position_length();
            self.write_arc(pos, to, Some(&arc_label), None);
            last_end_pos = to;
        }
        ts.end()?;
        if last_end_pos != -1 {
            self.write_node(-2, None);
            self.write_arc(last_end_pos, -2, None, None);
        }
        self.out.push_str("}\n");
        Ok(self.out)
    }

    fn write_arc(&mut self, from: i32, to: i32, label: Option<&str>, style: Option<&str>) {
        let _ = write!(self.out, "  {from} -> {to} [");
        if let Some(label) = label {
            let _ = write!(self.out, " label=\"{label}\"");
        }
        if let Some(style) = style {
            let _ = write!(self.out, " style=\"{style}\"");
        }
        self.out.push_str("]\n");
    }

    fn write_node(&mut self, name: i32, label: Option<&str>) {
        let _ = write!(self.out, "  {name}");
        match label {
            Some(label) => {
                let _ = write!(self.out, " [label=\"{label}\"]");
            }
            None => self.out.push_str(" [shape=point color=white]"),
        }
        self.out.push('\n');
    }

    fn write_header(&mut self) {
        self.out.push_str("digraph tokens {\n");
        self.out.push_str("  graph [ fontsize=30 labelloc=\"t\" label=\"\" splines=true overlap=false rankdir = \"LR\" ];\n");
        self.out.push_str("  // A2 paper size\n");
        self.out.push_str("  size = \"34.4,16.5\";\n");
        let _ = writeln!(
            self.out,
            "  edge [ fontname=\"{FONT_NAME}\" fontcolor=\"red\" color=\"#606060\" ]"
        );
        let _ = writeln!(
            self.out,
            "  node [ style=\"filled\" fillcolor=\"#e8e8f0\" shape=\"Mrecord\" fontname=\"{FONT_NAME}\" ]"
        );
        self.out.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::canned::CannedTokenStream;
    use crate::analysis::token::Token;

    fn stream() -> CannedTokenStream {
        CannedTokenStream::new(vec![
            Token::with_pos_inc("a", 0, 0, 1),
            Token::with_pos_len("ab", 0, 0, 3, 2),
            Token::with_pos_inc("C", 3, 2, 3),
        ])
    }

    #[test]
    fn dot_with_and_without_text() {
        let dot = TokenStreamToDot::new(Some("a c"))
            .to_dot(&mut stream())
            .unwrap();
        assert!(dot.starts_with("digraph tokens {\n"), "{dot}");
        assert!(dot.contains("  0 [label=\"0\"]\n  -1 [shape=point color=white]\n  -1 -> 0 []\n  0 -> 1 [ label=\"a\"]\n"), "{dot}");
        assert!(dot.contains("  0 -> 2 [ label=\"ab / a c\"]\n"), "{dot}");
        assert!(dot.contains("  2 -> 3 [ style=\"dotted\"]\n"), "{dot}");
        assert!(dot.contains("  3 -> 4 [ label=\"C / c\"]\n"), "{dot}");
        assert!(
            dot.ends_with("  -2 [shape=point color=white]\n  4 -> -2 []\n}\n"),
            "{dot}"
        );
        let dot = TokenStreamToDot::new(None).to_dot(&mut stream()).unwrap();
        assert!(dot.contains("label=\"a / 0-1\""), "{dot}");
        let empty = TokenStreamToDot::new(None)
            .to_dot(&mut CannedTokenStream::new(vec![]))
            .unwrap();
        assert!(empty.ends_with("\n\n}\n"), "{empty}");
    }

    #[test]
    fn offsets_past_the_text_are_an_error() {
        let e = TokenStreamToDot::new(Some(""))
            .to_dot(&mut stream())
            .unwrap_err();
        assert!(matches!(e, AnalysisError::IllegalArgument(_)));
    }
}
