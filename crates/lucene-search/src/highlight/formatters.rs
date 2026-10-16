//! The small parts of `org.apache.lucene.search.highlight`: the formatters
//! (`SimpleHTMLFormatter`, `GradientFormatter`, `SpanGradientFormatter`),
//! the encoders (`DefaultEncoder`, `SimpleHTMLEncoder`) and the fragmenters
//! (`SimpleFragmenter`, `NullFragmenter`, `SimpleSpanFragmenter`).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use lucene_analysis::AttributeSource;

use super::{Encoder, Error, Formatter, Fragmenter, Result, TokenGroup, WeightedSpanTerm};

/// `SimpleHTMLFormatter`.
#[derive(Debug, Clone)]
pub struct SimpleHtmlFormatter {
    pre_tag: String,
    post_tag: String,
}

impl Default for SimpleHtmlFormatter {
    /// `new SimpleHTMLFormatter()`: `<B>` and `</B>`.
    fn default() -> Self {
        Self::new("<B>", "</B>")
    }
}

impl SimpleHtmlFormatter {
    /// `new SimpleHTMLFormatter(preTag, postTag)`.
    pub fn new(pre_tag: impl Into<String>, post_tag: impl Into<String>) -> Self {
        SimpleHtmlFormatter {
            pre_tag: pre_tag.into(),
            post_tag: post_tag.into(),
        }
    }
}

impl Formatter for SimpleHtmlFormatter {
    fn highlight_term(&self, original_text: &str, token_group: &TokenGroup) -> String {
        if token_group.total_score() <= 0.0 {
            return original_text.to_string();
        }
        format!("{}{}{}", self.pre_tag, original_text, self.post_tag)
    }
}

/// `GradientFormatter`: a `<font>` whose colours grade with the group's score.
#[derive(Debug, Clone)]
pub struct GradientFormatter {
    max_score: f32,
    fg_min: [i32; 3],
    fg_max: [i32; 3],
    highlight_foreground: bool,
    bg_min: [i32; 3],
    bg_max: [i32; 3],
    highlight_background: bool,
    span: bool,
}

/// `hexToInt(hex)`.
fn hex_to_int(hex: &str) -> Result<i32> {
    let mut l: i64 = 0;
    if hex.chars().count() > 16 {
        return Err(Error::IllegalArgument("NumberFormatException".into()));
    }
    for c in hex.chars() {
        let d = c
            .to_digit(16)
            .ok_or_else(|| Error::IllegalArgument("NumberFormatException".into()))?;
        l = (l << 4) | i64::from(d);
    }
    Ok(i32::try_from(l).unwrap_or(i32::MAX))
}

fn parse_colour(colour: &str, what: &str) -> Result<[i32; 3]> {
    if colour.chars().count() != 7 {
        return Err(Error::IllegalArgument(format!(
            "{what} is not 7 bytes long eg a hex RGB value such as #FFFFFF"
        )));
    }
    let units: Vec<char> = colour.chars().collect();
    let part = |a: usize, b: usize| hex_to_int(&units[a..b].iter().collect::<String>());
    Ok([part(1, 3)?, part(3, 5)?, part(5, 7)?])
}

impl GradientFormatter {
    /// `new GradientFormatter(maxScore, minForegroundColor, maxForegroundColor,
    /// minBackgroundColor, maxBackgroundColor)`.
    pub fn new(
        max_score: f32,
        min_foreground: Option<&str>,
        max_foreground: Option<&str>,
        min_background: Option<&str>,
        max_background: Option<&str>,
    ) -> Result<Self> {
        let mut f = GradientFormatter {
            max_score,
            fg_min: [0; 3],
            fg_max: [0; 3],
            highlight_foreground: false,
            bg_min: [0; 3],
            bg_max: [0; 3],
            highlight_background: false,
            span: false,
        };
        if let (Some(min), Some(max)) = (min_foreground, max_foreground) {
            f.highlight_foreground = true;
            // Java's second message names `minForegroundColor` for both
            f.fg_min = parse_colour(min, "minForegroundColor")?;
            f.fg_max = parse_colour(max, "minForegroundColor")?;
        }
        if let (Some(min), Some(max)) = (min_background, max_background) {
            f.highlight_background = true;
            f.bg_min = parse_colour(min, "minBackgroundColor")?;
            f.bg_max = parse_colour(max, "minBackgroundColor")?;
        }
        Ok(f)
    }

    fn colour_val(&self, min: i32, max: i32, score: f32) -> i32 {
        if min == max {
            return min;
        }
        let scale = (min - max).abs() as f32;
        let rel_score_percent = self.max_score.min(score) / self.max_score;
        let col_score = scale * rel_score_percent;
        min.min(max) + col_score as i32
    }

    fn colour_string(&self, min: [i32; 3], max: [i32; 3], score: f32) -> String {
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        let mut s = String::from("#");
        for i in 0..3 {
            let v = self.colour_val(min[i], max[i], score);
            s.push(char::from(
                HEX[usize::try_from((v & 0xF0) >> 4).unwrap_or(0)],
            ));
            s.push(char::from(HEX[usize::try_from(v & 0x0F).unwrap_or(0)]));
        }
        s
    }

    fn foreground(&self, score: f32) -> String {
        self.colour_string(self.fg_min, self.fg_max, score)
    }

    fn background(&self, score: f32) -> String {
        self.colour_string(self.bg_min, self.bg_max, score)
    }
}

impl Formatter for GradientFormatter {
    fn highlight_term(&self, original_text: &str, token_group: &TokenGroup) -> String {
        let score = token_group.total_score();
        if score == 0.0 {
            return original_text.to_string();
        }
        let mut sb = String::new();
        if self.span {
            sb.push_str("<span style=\"");
            if self.highlight_foreground {
                sb.push_str("color: ");
                sb.push_str(&self.foreground(score));
                sb.push_str("; ");
            }
            if self.highlight_background {
                sb.push_str("background: ");
                sb.push_str(&self.background(score));
                sb.push_str("; ");
            }
            sb.push_str("\">");
            sb.push_str(original_text);
            sb.push_str("</span>");
            return sb;
        }
        sb.push_str("<font ");
        if self.highlight_foreground {
            sb.push_str("color=\"");
            sb.push_str(&self.foreground(score));
            sb.push_str("\" ");
        }
        if self.highlight_background {
            sb.push_str("bgcolor=\"");
            sb.push_str(&self.background(score));
            sb.push_str("\" ");
        }
        sb.push('>');
        sb.push_str(original_text);
        sb.push_str("</font>");
        sb
    }
}

/// `SpanGradientFormatter`: the same grades in a `<span style>`.
#[derive(Debug, Clone)]
pub struct SpanGradientFormatter(GradientFormatter);

impl SpanGradientFormatter {
    /// `new SpanGradientFormatter(maxScore, minForegroundColor,
    /// maxForegroundColor, minBackgroundColor, maxBackgroundColor)`.
    pub fn new(
        max_score: f32,
        min_foreground: Option<&str>,
        max_foreground: Option<&str>,
        min_background: Option<&str>,
        max_background: Option<&str>,
    ) -> Result<Self> {
        let mut f = GradientFormatter::new(
            max_score,
            min_foreground,
            max_foreground,
            min_background,
            max_background,
        )?;
        f.span = true;
        Ok(SpanGradientFormatter(f))
    }
}

impl Formatter for SpanGradientFormatter {
    fn highlight_term(&self, original_text: &str, token_group: &TokenGroup) -> String {
        self.0.highlight_term(original_text, token_group)
    }
}

/// `DefaultEncoder`: the text as it is.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultEncoder;

impl Encoder for DefaultEncoder {
    fn encode_text(&self, original_text: &str) -> String {
        original_text.to_string()
    }
}

/// `SimpleHTMLEncoder`: `"` `&` `<` `>` `'` and `/` as entities.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleHtmlEncoder;

impl SimpleHtmlEncoder {
    /// `htmlEncode(plainText)`.
    pub fn html_encode(plain_text: &str) -> String {
        let mut result = String::with_capacity(plain_text.len());
        for ch in plain_text.chars() {
            match ch {
                '"' => result.push_str("&quot;"),
                '&' => result.push_str("&amp;"),
                '<' => result.push_str("&lt;"),
                '>' => result.push_str("&gt;"),
                '\'' => result.push_str("&#x27;"),
                '/' => result.push_str("&#x2F;"),
                c => result.push(c),
            }
        }
        result
    }
}

impl Encoder for SimpleHtmlEncoder {
    fn encode_text(&self, original_text: &str) -> String {
        Self::html_encode(original_text)
    }
}

/// `SimpleFragmenter`: a new fragment every `fragment_size` characters.
#[derive(Debug, Clone)]
pub struct SimpleFragmenter {
    current_num_frags: i32,
    fragment_size: i32,
}

impl Default for SimpleFragmenter {
    /// `new SimpleFragmenter()`: 100 characters.
    fn default() -> Self {
        Self::new(100)
    }
}

impl SimpleFragmenter {
    /// `new SimpleFragmenter(fragmentSize)`.
    pub fn new(fragment_size: i32) -> Self {
        SimpleFragmenter {
            current_num_frags: 1,
            fragment_size,
        }
    }

    /// `getFragmentSize()`.
    pub fn fragment_size(&self) -> i32 {
        self.fragment_size
    }

    /// `setFragmentSize(size)`.
    pub fn set_fragment_size(&mut self, size: i32) {
        self.fragment_size = size;
    }
}

impl Fragmenter for SimpleFragmenter {
    fn start(&mut self, _original_text: &str) {
        self.current_num_frags = 1;
    }

    fn is_new_fragment(&mut self, token: &AttributeSource) -> bool {
        let is_new = token.end_offset() >= self.fragment_size.wrapping_mul(self.current_num_frags);
        if is_new {
            self.current_num_frags += 1;
        }
        is_new
    }
}

/// `NullFragmenter`: one fragment, the whole text.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullFragmenter;

impl Fragmenter for NullFragmenter {
    fn start(&mut self, _original_text: &str) {}

    fn is_new_fragment(&mut self, _token: &AttributeSource) -> bool {
        false
    }
}

/// `SimpleSpanFragmenter`: like [`SimpleFragmenter`], but never cuts inside a
/// span the query matched.
pub struct SimpleSpanFragmenter {
    fragment_size: i32,
    current_num_frags: i32,
    position: i32,
    terms: Arc<RwLock<HashMap<String, WeightedSpanTerm>>>,
    wait_for_pos: i32,
    text_size: i32,
}

impl SimpleSpanFragmenter {
    /// `new SimpleSpanFragmenter(queryScorer)`: 100 characters.
    pub fn new(scorer: &super::QueryScorer) -> Self {
        Self::with_size(scorer, 100)
    }

    /// `new SimpleSpanFragmenter(queryScorer, fragmentSize)`.
    pub fn with_size(scorer: &super::QueryScorer, fragment_size: i32) -> Self {
        SimpleSpanFragmenter {
            fragment_size,
            current_num_frags: 0,
            position: -1,
            terms: scorer.shared_terms(),
            wait_for_pos: -1,
            text_size: 0,
        }
    }
}

impl Fragmenter for SimpleSpanFragmenter {
    fn start(&mut self, original_text: &str) {
        self.position = -1;
        self.current_num_frags = 1;
        self.text_size = i32::try_from(original_text.encode_utf16().count()).unwrap_or(i32::MAX);
    }

    fn is_new_fragment(&mut self, token: &AttributeSource) -> bool {
        self.position += token.position_increment();
        if self.wait_for_pos <= self.position {
            self.wait_for_pos = -1;
        } else if self.wait_for_pos != -1 {
            return false;
        }
        let terms = self.terms.read().unwrap_or_else(|e| e.into_inner());
        if let Some(w) = terms.get(token.term()) {
            for span in w.position_spans() {
                if span.start() == self.position {
                    self.wait_for_pos = span.end() + 1;
                    break;
                }
            }
        }
        let is_new = token.end_offset() >= self.fragment_size.wrapping_mul(self.current_num_frags)
            && (self.text_size - token.end_offset()) >= (self.fragment_size >> 1);
        if is_new {
            self.current_num_frags += 1;
        }
        is_new
    }
}
