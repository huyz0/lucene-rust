//! `WKTReader` (`org.locationtech.spatial4j.io`): POINT, MULTIPOINT,
//! ENVELOPE, LINESTRING, MULTILINESTRING, POLYGON, MULTIPOLYGON,
//! GEOMETRYCOLLECTION and Spatial4j's BUFFER, with EMPTY and dimension
//! tokens (Z, M, ...) accepted and extra ordinates ignored.
//!
//! Offsets in [`Error::Parse`] count UTF-16 units, as Java's do, for the
//! Basic Multilingual Plane (characters past it count once here).

use std::sync::Arc;

use super::context::SpatialContext;
use super::shape::Shape;
use super::shape_factory::PointsBuilder;
use super::{Error, Result};

/// `WKTReader`.
#[derive(Debug, Clone)]
pub struct WktReader {
    ctx: Arc<SpatialContext>,
}

/// `Character.isWhitespace(c)`.
pub(crate) fn java_is_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | '\u{1C}' | '\u{1D}' | '\u{1E}' | '\u{1F}'
    ) || (c.is_whitespace()
        && !matches!(c, '\u{A0}' | '\u{2007}' | '\u{202F}' | '\u{85}')
        && !c.is_ascii_control())
}

/// `Character.isJavaIdentifierPart(c)`.
fn java_is_identifier_part(c: char) -> bool {
    c.is_alphanumeric()
        || c == '$'
        || c == '_'
        || matches!(c, '\u{00}'..='\u{08}' | '\u{0E}'..='\u{1B}' | '\u{7F}'..='\u{9F}')
}

/// `Character.isDigit(c)`.
fn java_is_digit(c: char) -> bool {
    c.is_ascii_digit() || (!c.is_ascii() && c.is_numeric())
}

/// `Double.parseDouble(s)` for what `skipDouble` admits (digits, `.`, signs
/// and a non-leading exponent marker), with Java's
/// `NumberFormatException` message.
pub(crate) fn java_parse_double(s: &str) -> std::result::Result<f64, String> {
    let ok = !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'
        });
    if ok {
        if let Ok(v) = s.parse::<f64>() {
            return Ok(v);
        }
    }
    if s.is_empty() {
        return Err("java.lang.NumberFormatException: empty String".into());
    }
    Err(format!(
        "java.lang.NumberFormatException: For input string: \"{s}\""
    ))
}

/// `WKTReader.State`: the string and the offset of its next character.
#[derive(Debug, Clone)]
pub struct State {
    pub(crate) raw: Vec<char>,
    /// `offset`.
    pub offset: usize,
    /// `dimension`: a dimension token (`Z`, `M`, ...) after a shape name.
    pub dimension: Option<String>,
}

fn parse_err(message: impl Into<String>, offset: usize) -> Error {
    Error::Parse {
        message: message.into(),
        offset: offset as i32,
    }
}

impl State {
    fn new(raw: &str) -> Self {
        State {
            raw: raw.chars().collect(),
            offset: 0,
            dimension: None,
        }
    }

    fn char_at(&self, i: usize) -> char {
        self.raw[i]
    }

    /// `nextWord()`.
    pub fn next_word(&mut self) -> Result<String> {
        let start = self.offset;
        while self.offset < self.raw.len() && java_is_identifier_part(self.raw[self.offset]) {
            self.offset += 1;
        }
        if start == self.offset {
            return Err(parse_err("Word expected", start));
        }
        let result: String = self.raw[start..self.offset].iter().collect();
        self.next_if_whitespace();
        Ok(result)
    }

    /// `nextIfEmptyAndSkipZM()`.
    pub fn next_if_empty_and_skip_zm(&mut self) -> Result<bool> {
        if self.eof() {
            return Ok(false);
        }
        let c = self.char_at(self.offset);
        if c == '(' || !java_is_identifier_part(c) {
            return Ok(false);
        }
        let word = self.next_word()?;
        if word.eq_ignore_ascii_case("EMPTY") {
            return Ok(true);
        }
        self.dimension = Some(word);
        if self.eof() {
            return Ok(false);
        }
        let c = self.char_at(self.offset);
        if c == '(' || !java_is_identifier_part(c) {
            return Ok(false);
        }
        let word = self.next_word()?;
        if word.eq_ignore_ascii_case("EMPTY") {
            return Ok(true);
        }
        Err(parse_err(
            format!("Expected EMPTY because found dimension; but got [{word}]"),
            self.offset,
        ))
    }

    /// `nextDouble()`.
    pub fn next_double(&mut self) -> Result<f64> {
        let start = self.offset;
        self.skip_double();
        if start == self.offset {
            return Err(parse_err("Expected a number", self.offset));
        }
        let s: String = self.raw[start..self.offset].iter().collect();
        let result = java_parse_double(&s).map_err(|m| parse_err(m, self.offset))?;
        self.next_if_whitespace();
        Ok(result)
    }

    /// `skipDouble()`.
    pub fn skip_double(&mut self) {
        let start = self.offset;
        while self.offset < self.raw.len() {
            let c = self.raw[self.offset];
            if !(java_is_digit(c) || c == '.' || c == '-' || c == '+') {
                // 'e' is okay as long as it isn't first
                if self.offset != start && (c == 'e' || c == 'E') {
                    self.offset += 1;
                    continue;
                }
                break;
            }
            self.offset += 1;
        }
    }

    /// `skipNextDoubles()`.
    pub fn skip_next_doubles(&mut self) {
        while !self.eof() {
            let start = self.offset;
            self.skip_double();
            if start == self.offset {
                return;
            }
            self.next_if_whitespace();
        }
    }

    /// `nextExpect(expected)`.
    pub fn next_expect(&mut self, expected: char) -> Result<()> {
        if self.eof() {
            return Err(parse_err(
                format!("Expected [{expected}] found EOF"),
                self.offset,
            ));
        }
        let c = self.char_at(self.offset);
        if c != expected {
            return Err(parse_err(
                format!("Expected [{expected}] found [{c}]"),
                self.offset,
            ));
        }
        self.offset += 1;
        self.next_if_whitespace();
        Ok(())
    }

    /// `eof()`.
    pub fn eof(&self) -> bool {
        self.offset >= self.raw.len()
    }

    /// `nextIf(expected)`.
    pub fn next_if(&mut self, expected: char) -> bool {
        if !self.eof() && self.char_at(self.offset) == expected {
            self.offset += 1;
            self.next_if_whitespace();
            return true;
        }
        false
    }

    /// `nextIfWhitespace()`.
    pub fn next_if_whitespace(&mut self) {
        while self.offset < self.raw.len() && java_is_whitespace(self.raw[self.offset]) {
            self.offset += 1;
        }
    }

    /// `nextSubShapeString()`.
    pub fn next_sub_shape_string(&mut self) -> Result<String> {
        let start = self.offset;
        let mut paren_stack = 0i32;
        while self.offset < self.raw.len() {
            let c = self.raw[self.offset];
            if c == ',' {
                if paren_stack == 0 {
                    break;
                }
            } else if c == ')' {
                if paren_stack == 0 {
                    break;
                }
                paren_stack -= 1;
            } else if c == '(' {
                paren_stack += 1;
            }
            self.offset += 1;
        }
        if paren_stack != 0 {
            return Err(parse_err("Unbalanced parenthesis", start));
        }
        Ok(self.raw[start..self.offset].iter().collect())
    }
}

impl WktReader {
    /// `new WKTReader(ctx, factory)`.
    pub fn new(ctx: Arc<SpatialContext>) -> Self {
        WktReader { ctx }
    }

    /// `parse(wktString)`.
    pub fn parse(&self, wkt: &str) -> Result<Arc<dyn Shape>> {
        if let Some(shape) = self.parse_if_supported(wkt)? {
            return Ok(shape);
        }
        let shortened: String = if wkt.encode_utf16().count() <= 128 {
            wkt.to_string()
        } else {
            let head: Vec<u16> = wkt.encode_utf16().take(125).collect();
            format!("{}...", String::from_utf16_lossy(&head))
        };
        Err(parse_err(
            format!("Unknown Shape definition [{shortened}]"),
            0,
        ))
    }

    /// `parseIfSupported(wktString)`: `None` for an unknown shape name or a
    /// blank string.
    pub fn parse_if_supported(&self, wkt: &str) -> Result<Option<Arc<dyn Shape>>> {
        let mut state = State::new(wkt);
        state.next_if_whitespace();
        if state.eof() {
            return Ok(None);
        }
        if !state.char_at(state.offset).is_alphabetic() {
            return Ok(None);
        }
        let shape_type = state.next_word()?;
        let result = match self.parse_shape_by_type(&mut state, &shape_type) {
            Ok(r) => r,
            Err(e @ (Error::Parse { .. } | Error::InvalidShape(_))) => return Err(e),
            Err(Error::IllegalArgument(m)) => return Err(Error::InvalidShape(m)),
            Err(Error::Spatial3d(crate::spatial3d::Error::IllegalArgument(m))) => {
                return Err(Error::InvalidShape(m))
            }
            Err(e) => return Err(parse_err(e.java_to_string(), state.offset)),
        };
        if result.is_some() && !state.eof() {
            return Err(parse_err("end of shape expected", state.offset));
        }
        Ok(result)
    }

    /// `parseShapeByType(state, shapeType)`.
    pub fn parse_shape_by_type(
        &self,
        state: &mut State,
        shape_type: &str,
    ) -> Result<Option<Arc<dyn Shape>>> {
        let t = |name: &str| shape_type.eq_ignore_ascii_case(name);
        let shape = if t("POINT") {
            self.parse_point_shape(state)?
        } else if t("MULTIPOINT") {
            self.parse_multi_point_shape(state)?
        } else if t("ENVELOPE") {
            self.parse_envelope_shape(state)?
        } else if t("LINESTRING") {
            self.parse_line_string_shape(state)?
        } else if t("POLYGON") {
            self.parse_polygon_shape(state)?
        } else if t("GEOMETRYCOLLECTION") {
            self.parse_geometry_collection_shape(state)?
        } else if t("MULTILINESTRING") {
            self.parse_multi_line_string_shape(state)?
        } else if t("MULTIPOLYGON") {
            self.parse_multi_polygon_shape(state)?
        } else if t("BUFFER") {
            self.parse_buffer_shape(state)?
        } else {
            return Ok(None);
        };
        Ok(Some(shape))
    }

    /// `parseBufferShape(state)`.
    fn parse_buffer_shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        state.next_expect('(')?;
        let shape = self.shape(state)?;
        state.next_expect(',')?;
        let distance = self.ctx.shape_factory().norm_dist(state.next_double()?);
        state.next_expect(')')?;
        shape.buffered(distance, &self.ctx)
    }

    /// `parsePointShape(state)`.
    fn parse_point_shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        if state.next_if_empty_and_skip_zm()? {
            return Ok(self.ctx.point_xy(f64::NAN, f64::NAN)?);
        }
        state.next_expect('(')?;
        let mut one = OnePointsBuilder {
            ctx: &self.ctx,
            point: None,
        };
        self.point(state, &mut one)?;
        state.next_expect(')')?;
        Ok(one.point.expect("a point was read"))
    }

    /// `parseMultiPointShape(state)`.
    fn parse_multi_point_shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        let mut builder = self.ctx.shape_factory().multi_point_builder(&self.ctx);
        if state.next_if_empty_and_skip_zm()? {
            return builder.build();
        }
        state.next_expect('(')?;
        loop {
            let open_paren = state.next_if('(');
            self.point(state, builder.as_points_builder())?;
            if open_paren {
                state.next_expect(')')?;
            }
            if !state.next_if(',') {
                break;
            }
        }
        state.next_expect(')')?;
        builder.build()
    }

    /// `parseEnvelopeShape(state)`: `(x1, x2, y2, y1)`.
    fn parse_envelope_shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        state.next_expect('(')?;
        let x1 = state.next_double()?;
        state.next_expect(',')?;
        let x2 = state.next_double()?;
        state.next_expect(',')?;
        let y2 = state.next_double()?;
        state.next_expect(',')?;
        let y1 = state.next_double()?;
        state.next_expect(')')?;
        let f = self.ctx.shape_factory();
        Ok(self
            .ctx
            .rect(f.norm_x(x1), f.norm_x(x2), f.norm_y(y1), f.norm_y(y2))?)
    }

    /// `parseLineStringShape(state)`.
    fn parse_line_string_shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        let mut builder = self.ctx.shape_factory().line_string_builder(&self.ctx);
        if state.next_if_empty_and_skip_zm()? {
            return builder.build();
        }
        self.point_list(state, builder.as_points_builder())?;
        builder.build()
    }

    /// `parseMultiLineStringShape(state)`.
    fn parse_multi_line_string_shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        let mut multi = self
            .ctx
            .shape_factory()
            .multi_line_string_builder(&self.ctx);
        if !state.next_if_empty_and_skip_zm()? {
            state.next_expect('(')?;
            loop {
                let mut ls = multi.line_string();
                self.point_list(state, ls.as_points_builder())?;
                multi.add(ls)?;
                if !state.next_if(',') {
                    break;
                }
            }
            state.next_expect(')')?;
        }
        multi.build()
    }

    /// `parsePolygonShape(state)`.
    fn parse_polygon_shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        let mut builder = self.ctx.shape_factory().polygon_builder(&self.ctx)?;
        if !state.next_if_empty_and_skip_zm()? {
            self.polygon(state, &mut *builder)?;
        }
        builder.build_or_rect()
    }

    /// `parseMulitPolygonShape(state)`.
    fn parse_multi_polygon_shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        let mut multi = self.ctx.shape_factory().multi_polygon_builder(&self.ctx);
        if !state.next_if_empty_and_skip_zm()? {
            state.next_expect('(')?;
            loop {
                let mut poly = multi.polygon()?;
                self.polygon(state, &mut *poly)?;
                multi.add(poly)?;
                if !state.next_if(',') {
                    break;
                }
            }
            state.next_expect(')')?;
        }
        multi.build()
    }

    /// `parseGeometryCollectionShape(state)`.
    fn parse_geometry_collection_shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        let mut builder = self.ctx.shape_factory().multi_shape_builder(&self.ctx);
        if state.next_if_empty_and_skip_zm()? {
            return builder.build();
        }
        state.next_expect('(')?;
        loop {
            let shape = self.shape(state)?;
            builder.add(shape)?;
            if !state.next_if(',') {
                break;
            }
        }
        state.next_expect(')')?;
        builder.build()
    }

    /// `shape(state)`: a nested shape, which must be known.
    fn shape(&self, state: &mut State) -> Result<Arc<dyn Shape>> {
        let ty = state.next_word()?;
        match self.parse_shape_by_type(state, &ty)? {
            Some(s) => Ok(s),
            None => Err(parse_err(
                format!("Shape of type {ty} is unknown"),
                state.offset,
            )),
        }
    }

    /// `pointList(state, pointsBuilder)`.
    fn point_list(&self, state: &mut State, builder: &mut dyn PointsBuilder) -> Result<()> {
        state.next_expect('(')?;
        loop {
            self.point(state, builder)?;
            if !state.next_if(',') {
                break;
            }
        }
        state.next_expect(')')
    }

    /// `point(state, pointsBuilder)`: the first two numbers, normalised.
    fn point(&self, state: &mut State, builder: &mut dyn PointsBuilder) -> Result<()> {
        let x = state.next_double()?;
        let y = state.next_double()?;
        state.skip_next_doubles();
        let f = self.ctx.shape_factory();
        builder.point_xy(f.norm_x(x), f.norm_y(y))
    }

    /// `polygon(state, polygonBuilder)`: the outer ring, then the holes.
    fn polygon(
        &self,
        state: &mut State,
        builder: &mut dyn super::shape_factory::PolygonBuilder,
    ) -> Result<()> {
        state.next_expect('(')?;
        self.point_list(state, builder.as_points_builder())?;
        while state.next_if(',') {
            let mut hole = builder.hole();
            self.point_list(state, hole.as_points_builder())?;
            hole.end_hole()?;
        }
        state.next_expect(')')
    }
}

/// `OnePointsBuilder`: keeps the one point a POINT has.
struct OnePointsBuilder<'a> {
    ctx: &'a Arc<SpatialContext>,
    point: Option<Arc<dyn Shape>>,
}

impl PointsBuilder for OnePointsBuilder<'_> {
    fn point_xy(&mut self, x: f64, y: f64) -> Result<()> {
        self.point = Some(self.ctx.point_xy(x, y)?);
        Ok(())
    }

    fn point_xyz(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        self.point = Some(self.ctx.shape_factory().point_xyz(self.ctx, x, y, z)?);
        Ok(())
    }
}
