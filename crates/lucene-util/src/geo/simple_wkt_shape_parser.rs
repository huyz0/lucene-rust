//! Port of `org.apache.lucene.geo.SimpleWKTShapeParser`: parses WKT
//! (`POINT`, `MULTIPOINT`, `LINESTRING`, `MULTILINESTRING`, `POLYGON`,
//! `MULTIPOLYGON`, `GEOMETRYCOLLECTION`, and the non-standard `ENVELOPE`/
//! `BBOX`) into the lat/lon geometries.
//!
//! Java drives the parse with a `java.io.StreamTokenizer` in a particular
//! syntax (words of letters, digits, `.`, `+`, `-` and chars >= 160;
//! whitespace up to `' '`; `#` comments; everything else an ordinary
//! one-char token). [`StreamTokenizer`] here is a port of exactly that
//! configuration, including its line counting, which is what a
//! `ParseException`'s offset reports. It reads UTF-16 units, as Java's
//! `Reader` does. Numbers go through `Double.parseDouble`'s grammar
//! (`java_parse_double`), not Rust's.
//!
//! Java returns `Object`; here the result is a [`WktGeometry`], and an
//! `EMPTY` geometry (Java's `null`) is `None`.

use super::line::Line;
use super::polygon::Polygon;
use super::rectangle::Rectangle;
use super::{java_parse_double, GeoError};

/// `SimpleWKTShapeParser.EMPTY`.
pub const EMPTY: &str = "EMPTY";
/// `SPACE`.
pub const SPACE: &str = " ";
/// `LPAREN`.
pub const LPAREN: &str = "(";
/// `RPAREN`.
pub const RPAREN: &str = ")";
/// `COMMA`.
pub const COMMA: &str = ",";
/// `NAN`.
pub const NAN: &str = "NaN";

const NUMBER: &str = "<NUMBER>";
const EOF: &str = "END-OF-STREAM";
const EOL: &str = "END-OF-LINE";

/// `SimpleWKTShapeParser.ShapeType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeType {
    /// `POINT`.
    Point,
    /// `MULTIPOINT`.
    MultiPoint,
    /// `LINESTRING`.
    LineString,
    /// `MULTILINESTRING`.
    MultiLineString,
    /// `POLYGON`.
    Polygon,
    /// `MULTIPOLYGON`.
    MultiPolygon,
    /// `GEOMETRYCOLLECTION`.
    GeometryCollection,
    /// `ENVELOPE` (not part of the WKT spec; written `BBOX`).
    Envelope,
}

impl ShapeType {
    /// The enum constant's `name()`, which `toString()` prints.
    pub fn name(self) -> &'static str {
        match self {
            ShapeType::Point => "POINT",
            ShapeType::MultiPoint => "MULTIPOINT",
            ShapeType::LineString => "LINESTRING",
            ShapeType::MultiLineString => "MULTILINESTRING",
            ShapeType::Polygon => "POLYGON",
            ShapeType::MultiPolygon => "MULTIPOLYGON",
            ShapeType::GeometryCollection => "GEOMETRYCOLLECTION",
            ShapeType::Envelope => "ENVELOPE",
        }
    }

    /// `typename()`.
    fn typename(self) -> &'static str {
        match self {
            ShapeType::Point => "point",
            ShapeType::MultiPoint => "multipoint",
            ShapeType::LineString => "linestring",
            ShapeType::MultiLineString => "multilinestring",
            ShapeType::Polygon => "polygon",
            ShapeType::MultiPolygon => "multipolygon",
            ShapeType::GeometryCollection => "geometrycollection",
            ShapeType::Envelope => "envelope",
        }
    }

    /// `wktName()`.
    pub fn wkt_name(self) -> &'static str {
        if self == ShapeType::Envelope {
            "BBOX"
        } else {
            self.typename()
        }
    }

    /// `forName(shapename)`: case-insensitive, `bbox` meaning `ENVELOPE`.
    pub fn for_name(shapename: &str) -> Result<ShapeType, GeoError> {
        let typename = shapename.to_lowercase();
        const ALL: [ShapeType; 8] = [
            ShapeType::Point,
            ShapeType::MultiPoint,
            ShapeType::LineString,
            ShapeType::MultiLineString,
            ShapeType::Polygon,
            ShapeType::MultiPolygon,
            ShapeType::GeometryCollection,
            ShapeType::Envelope,
        ];
        if typename == "bbox" {
            return Ok(ShapeType::Envelope);
        }
        ALL.into_iter()
            .find(|t| t.typename() == typename)
            .ok_or_else(|| GeoError::illegal(format!("unknown geo_shape [{shapename}]")))
    }
}

/// What `parse` returns: Java's `Object` result, by type.
#[derive(Debug, Clone, PartialEq)]
pub enum WktGeometry {
    /// `POINT`: `double[] {x, y}` (lon, lat).
    Point([f64; 2]),
    /// `MULTIPOINT`: `double[][]` of `{lon, lat}`.
    MultiPoint(Vec<[f64; 2]>),
    /// `LINESTRING`.
    Line(Line),
    /// `MULTILINESTRING` (an `EMPTY` member is `None`).
    MultiLine(Vec<Option<Line>>),
    /// `POLYGON`.
    Polygon(Polygon),
    /// `MULTIPOLYGON` (an `EMPTY` member is `None`).
    MultiPolygon(Vec<Option<Polygon>>),
    /// `ENVELOPE`/`BBOX`.
    Envelope(Rectangle),
    /// `GEOMETRYCOLLECTION`.
    GeometryCollection(Vec<Option<WktGeometry>>),
}

/// `SimpleWKTShapeParser.parse(wkt)`.
pub fn parse(wkt: &str) -> Result<Option<WktGeometry>, GeoError> {
    parse_expected_type(wkt, None)
}

/// `SimpleWKTShapeParser.parseExpectedType(wkt, shapeType)`.
pub fn parse_expected_type(
    wkt: &str,
    shape_type: Option<ShapeType>,
) -> Result<Option<WktGeometry>, GeoError> {
    let mut stream = StreamTokenizer::new(wkt);
    let geometry = parse_geometry(&mut stream, shape_type)?;
    check_eof(&mut stream)?;
    Ok(geometry)
}

fn parse_error(stream: &StreamTokenizer, message: String) -> GeoError {
    GeoError::Parse {
        message,
        offset: stream.lineno,
    }
}

fn parse_geometry(
    stream: &mut StreamTokenizer,
    shape_type: Option<ShapeType>,
) -> Result<Option<WktGeometry>, GeoError> {
    let word = next_word(stream)?;
    let ty = ShapeType::for_name(&word)?;
    if let Some(expected) = shape_type {
        if expected != ShapeType::GeometryCollection && ty.wkt_name() != expected.wkt_name() {
            return Err(parse_error(
                stream,
                format!(
                    "Expected geometry type: [{}], but found: [{}]",
                    expected.name(),
                    ty.name()
                ),
            ));
        }
    }
    match ty {
        ShapeType::Point => Ok(parse_point(stream)?.map(WktGeometry::Point)),
        ShapeType::MultiPoint => Ok(parse_multi_point(stream)?.map(WktGeometry::MultiPoint)),
        ShapeType::LineString => Ok(parse_line(stream)?.map(WktGeometry::Line)),
        ShapeType::MultiLineString => Ok(parse_multi_line(stream)?.map(WktGeometry::MultiLine)),
        ShapeType::Polygon => Ok(parse_polygon(stream)?.map(WktGeometry::Polygon)),
        ShapeType::MultiPolygon => Ok(parse_multi_polygon(stream)?.map(WktGeometry::MultiPolygon)),
        ShapeType::Envelope => Ok(parse_bbox(stream)?.map(WktGeometry::Envelope)),
        ShapeType::GeometryCollection => {
            Ok(parse_geometry_collection(stream)?.map(WktGeometry::GeometryCollection))
        }
    }
}

/// `parsePoint`.
fn parse_point(stream: &mut StreamTokenizer) -> Result<Option<[f64; 2]>, GeoError> {
    if next_empty_or_open(stream)? == EMPTY {
        return Ok(None);
    }
    let pt = [next_number(stream)?, next_number(stream)?];
    if is_number_next(stream) {
        next_number(stream)?;
    }
    next_closer(stream)?;
    Ok(Some(pt))
}

/// `parseCoordinates`: a list of points into lats/lons.
fn parse_coordinates(
    stream: &mut StreamTokenizer,
    lats: &mut Vec<f64>,
    lons: &mut Vec<f64>,
) -> Result<(), GeoError> {
    let mut is_open_paren = false;
    if is_number_next(stream) || {
        is_open_paren = next_word(stream)? == LPAREN;
        is_open_paren
    } {
        parse_coordinate(stream, lats, lons)?;
    }
    while next_closer_or_comma(stream)? == COMMA {
        is_open_paren = false;
        if is_number_next(stream) || {
            is_open_paren = next_word(stream)? == LPAREN;
            is_open_paren
        } {
            parse_coordinate(stream, lats, lons)?;
        }
        if is_open_paren && next_closer(stream)? != RPAREN {
            return Err(parse_error(
                stream,
                format!("expected: [{RPAREN}] but found: [{}]", token_string(stream)),
            ));
        }
    }
    if is_open_paren && next_closer(stream)? != RPAREN {
        return Err(parse_error(
            stream,
            format!("expected: [{RPAREN}] but found: [{}]", token_string(stream)),
        ));
    }
    Ok(())
}

/// `parseCoordinate`: one coordinate, with an optional third dimension.
fn parse_coordinate(
    stream: &mut StreamTokenizer,
    lats: &mut Vec<f64>,
    lons: &mut Vec<f64>,
) -> Result<(), GeoError> {
    lons.push(next_number(stream)?);
    lats.push(next_number(stream)?);
    if is_number_next(stream) {
        next_number(stream)?;
    }
    Ok(())
}

fn parse_multi_point(stream: &mut StreamTokenizer) -> Result<Option<Vec<[f64; 2]>>, GeoError> {
    if next_empty_or_open(stream)? == EMPTY {
        return Ok(None);
    }
    let mut lats = Vec::new();
    let mut lons = Vec::new();
    parse_coordinates(stream, &mut lats, &mut lons)?;
    Ok(Some(
        lats.iter()
            .zip(&lons)
            .map(|(&lat, &lon)| [lon, lat])
            .collect(),
    ))
}

fn parse_line(stream: &mut StreamTokenizer) -> Result<Option<Line>, GeoError> {
    if next_empty_or_open(stream)? == EMPTY {
        return Ok(None);
    }
    let mut lats = Vec::new();
    let mut lons = Vec::new();
    parse_coordinates(stream, &mut lats, &mut lons)?;
    Ok(Some(Line::new(&lats, &lons)?))
}

fn parse_multi_line(stream: &mut StreamTokenizer) -> Result<Option<Vec<Option<Line>>>, GeoError> {
    if next_empty_or_open(stream)? == EMPTY {
        return Ok(None);
    }
    let mut lines = vec![parse_line(stream)?];
    while next_closer_or_comma(stream)? == COMMA {
        lines.push(parse_line(stream)?);
    }
    Ok(Some(lines))
}

fn parse_polygon_hole(stream: &mut StreamTokenizer) -> Result<Polygon, GeoError> {
    let mut lats = Vec::new();
    let mut lons = Vec::new();
    parse_coordinates(stream, &mut lats, &mut lons)?;
    Polygon::new(&lats, &lons, vec![])
}

fn parse_polygon(stream: &mut StreamTokenizer) -> Result<Option<Polygon>, GeoError> {
    if next_empty_or_open(stream)? == EMPTY {
        return Ok(None);
    }
    next_opener(stream)?;
    let mut lats = Vec::new();
    let mut lons = Vec::new();
    parse_coordinates(stream, &mut lats, &mut lons)?;
    let mut holes = Vec::new();
    while next_closer_or_comma(stream)? == COMMA {
        holes.push(parse_polygon_hole(stream)?);
    }
    Ok(Some(Polygon::new(&lats, &lons, holes)?))
}

fn parse_multi_polygon(
    stream: &mut StreamTokenizer,
) -> Result<Option<Vec<Option<Polygon>>>, GeoError> {
    if next_empty_or_open(stream)? == EMPTY {
        return Ok(None);
    }
    let mut polygons = vec![parse_polygon(stream)?];
    while next_closer_or_comma(stream)? == COMMA {
        polygons.push(parse_polygon(stream)?);
    }
    Ok(Some(polygons))
}

fn parse_bbox(stream: &mut StreamTokenizer) -> Result<Option<Rectangle>, GeoError> {
    if next_empty_or_open(stream)? == EMPTY {
        return Ok(None);
    }
    let min_lon = next_number(stream)?;
    next_comma(stream)?;
    let max_lon = next_number(stream)?;
    next_comma(stream)?;
    let max_lat = next_number(stream)?;
    next_comma(stream)?;
    let min_lat = next_number(stream)?;
    next_closer(stream)?;
    Ok(Some(Rectangle::new(min_lat, max_lat, min_lon, max_lon)?))
}

fn parse_geometry_collection(
    stream: &mut StreamTokenizer,
) -> Result<Option<Vec<Option<WktGeometry>>>, GeoError> {
    if next_empty_or_open(stream)? == EMPTY {
        return Ok(None);
    }
    let mut geometries = vec![parse_geometry(stream, Some(ShapeType::GeometryCollection))?];
    while next_closer_or_comma(stream)? == COMMA {
        geometries.push(parse_geometry(stream, None)?);
    }
    Ok(Some(geometries))
}

/// `nextWord`: a word, or one of `(`, `)`, `,`.
fn next_word(stream: &mut StreamTokenizer) -> Result<String, GeoError> {
    match stream.next_token() {
        TT_WORD => {
            let word = stream.sval.clone();
            Ok(if word.eq_ignore_ascii_case(EMPTY) {
                EMPTY.to_string()
            } else {
                word
            })
        }
        t if t == i32::from(b'(') => Ok(LPAREN.into()),
        t if t == i32::from(b')') => Ok(RPAREN.into()),
        t if t == i32::from(b',') => Ok(COMMA.into()),
        _ => Err(parse_error(
            stream,
            format!("expected word but found: {}", token_string(stream)),
        )),
    }
}

/// `nextNumber`.
fn next_number(stream: &mut StreamTokenizer) -> Result<f64, GeoError> {
    if stream.next_token() == TT_WORD {
        if stream.sval.eq_ignore_ascii_case(NAN) {
            return Ok(f64::NAN);
        }
        return java_parse_double(&stream.sval)
            .ok_or_else(|| parse_error(stream, format!("invalid number found: {}", stream.sval)));
    }
    Err(parse_error(
        stream,
        format!("expected number but found: {}", token_string(stream)),
    ))
}

/// `tokenString`.
fn token_string(stream: &StreamTokenizer) -> String {
    match stream.ttype {
        TT_WORD => stream.sval.clone(),
        TT_EOF => EOF.into(),
        TT_EOL => EOL.into(),
        TT_NUMBER => NUMBER.into(),
        c => format!("'{}'", String::from_utf16_lossy(&[c as u16])),
    }
}

/// `isNumberNext`: whether the next token is a word (peeked, pushed back).
fn is_number_next(stream: &mut StreamTokenizer) -> bool {
    let ty = stream.next_token();
    stream.push_back();
    ty == TT_WORD
}

fn next_empty_or_open(stream: &mut StreamTokenizer) -> Result<String, GeoError> {
    let next = next_word(stream)?;
    if next == EMPTY || next == LPAREN {
        return Ok(next);
    }
    Err(parse_error(
        stream,
        format!(
            "expected {EMPTY} or {LPAREN} but found: {}",
            token_string(stream)
        ),
    ))
}

fn next_closer(stream: &mut StreamTokenizer) -> Result<String, GeoError> {
    if next_word(stream)? == RPAREN {
        return Ok(RPAREN.into());
    }
    Err(parse_error(
        stream,
        format!("expected {RPAREN} but found: {}", token_string(stream)),
    ))
}

fn next_comma(stream: &mut StreamTokenizer) -> Result<String, GeoError> {
    if next_word(stream)? == COMMA {
        return Ok(COMMA.into());
    }
    Err(parse_error(
        stream,
        format!("expected {COMMA} but found: {}", token_string(stream)),
    ))
}

fn next_opener(stream: &mut StreamTokenizer) -> Result<String, GeoError> {
    if next_word(stream)? == LPAREN {
        return Ok(LPAREN.into());
    }
    Err(parse_error(
        stream,
        format!("expected {LPAREN} but found: {}", token_string(stream)),
    ))
}

fn next_closer_or_comma(stream: &mut StreamTokenizer) -> Result<String, GeoError> {
    let token = next_word(stream)?;
    if token == COMMA || token == RPAREN {
        return Ok(token);
    }
    Err(parse_error(
        stream,
        format!(
            "expected {COMMA} or {RPAREN} but found: {}",
            token_string(stream)
        ),
    ))
}

fn check_eof(stream: &mut StreamTokenizer) -> Result<(), GeoError> {
    if stream.next_token() != TT_EOF {
        return Err(parse_error(
            stream,
            format!(
                "expected end of WKT string but found additional text: {}",
                token_string(stream)
            ),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- tokenizer

const TT_EOF: i32 = -1;
const TT_EOL: i32 = b'\n' as i32;
const TT_NUMBER: i32 = -2;
const TT_WORD: i32 = -3;
const TT_NOTHING: i32 = -4;

const NEED_CHAR: i32 = i32::MAX;

const CT_WHITESPACE: u8 = 1;
const CT_ALPHA: u8 = 4;
const CT_COMMENT: u8 = 16;

/// `java.io.StreamTokenizer` configured as `parseExpectedType` configures
/// it: `resetSyntax()`, word chars `a-z A-Z 160-255 0-9 - + .`, whitespace
/// `0-' '`, comment char `#`. No quote chars, no number parsing, line ends
/// not significant, no slash comments.
struct StreamTokenizer {
    input: Vec<u16>,
    pos: usize,
    peekc: i32,
    pushed_back: bool,
    lineno: i32,
    ttype: i32,
    sval: String,
}

impl StreamTokenizer {
    fn new(s: &str) -> StreamTokenizer {
        StreamTokenizer {
            input: s.encode_utf16().collect(),
            pos: 0,
            peekc: NEED_CHAR,
            pushed_back: false,
            lineno: 1,
            ttype: TT_NOTHING,
            sval: String::new(),
        }
    }

    /// `Reader.read()`: the next UTF-16 unit.
    // SENTINEL: `-1` = end of input, as `java.io.Reader.read` defines it;
    // every caller tests `c < 0` before using the value as a char.
    fn read_char(&mut self) -> i32 {
        match self.input.get(self.pos) {
            Some(&c) => {
                self.pos += 1;
                i32::from(c)
            }
            None => -1,
        }
    }

    fn ctype(c: i32) -> u8 {
        if c >= 256 {
            return CT_ALPHA;
        }
        match c {
            0..=32 => CT_WHITESPACE,
            0x23 => CT_COMMENT, // '#'
            0x61..=0x7a | 0x41..=0x5a | 160..=255 | 0x30..=0x39 | 0x2d | 0x2b | 0x2e => CT_ALPHA,
            _ => 0,
        }
    }

    fn push_back(&mut self) {
        if self.ttype != TT_NOTHING {
            self.pushed_back = true;
        }
    }

    fn next_token(&mut self) -> i32 {
        if self.pushed_back {
            self.pushed_back = false;
            return self.ttype;
        }
        self.sval.clear();
        let mut c = self.peekc;
        if c < 0 {
            c = NEED_CHAR;
        }
        if c == NEED_CHAR {
            c = self.read_char();
            if c < 0 {
                self.ttype = TT_EOF;
                return TT_EOF;
            }
        }
        self.ttype = c;
        self.peekc = NEED_CHAR;
        let mut ctype = Self::ctype(c);
        while ctype & CT_WHITESPACE != 0 {
            if c == i32::from(b'\r') {
                self.lineno += 1;
                c = self.read_char();
                if c == i32::from(b'\n') {
                    c = self.read_char();
                }
            } else {
                if c == i32::from(b'\n') {
                    self.lineno += 1;
                }
                c = self.read_char();
            }
            if c < 0 {
                self.ttype = TT_EOF;
                return TT_EOF;
            }
            ctype = Self::ctype(c);
        }
        if ctype & CT_ALPHA != 0 {
            let mut buf: Vec<u16> = Vec::new();
            loop {
                buf.push(c as u16);
                c = self.read_char();
                let ct = if c < 0 { CT_WHITESPACE } else { Self::ctype(c) };
                if ct & CT_ALPHA == 0 {
                    break;
                }
            }
            self.peekc = c;
            self.sval = String::from_utf16_lossy(&buf);
            self.ttype = TT_WORD;
            return TT_WORD;
        }
        if ctype & CT_COMMENT != 0 {
            loop {
                c = self.read_char();
                if c == i32::from(b'\n') || c == i32::from(b'\r') || c < 0 {
                    break;
                }
            }
            self.peekc = c;
            return self.next_token();
        }
        self.ttype = c;
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        assert_eq!(
            parse("POINT (1 2)").unwrap(),
            Some(WktGeometry::Point([1.0, 2.0]))
        );
        assert_eq!(parse("point EMPTY").unwrap(), None);
        assert_eq!(
            parse("POINT (1 2 3)").unwrap(),
            Some(WktGeometry::Point([1.0, 2.0]))
        );
        assert_eq!(
            parse("MULTIPOINT (1 2, 3 4)").unwrap(),
            Some(WktGeometry::MultiPoint(vec![[1.0, 2.0], [3.0, 4.0]]))
        );
        assert_eq!(
            parse("BBOX (1, 2, 4, 3)").unwrap(),
            Some(WktGeometry::Envelope(
                Rectangle::new(3.0, 4.0, 1.0, 2.0).unwrap()
            ))
        );
        match parse("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING EMPTY)").unwrap() {
            Some(WktGeometry::GeometryCollection(g)) => {
                assert_eq!(g.len(), 2);
                assert_eq!(g[1], None);
            }
            other => panic!("{other:?}"),
        }
        let p =
            parse("POLYGON ((0 0, 1 0, 1 1, 0 0), (0.1 0.1, 0.2 0.1, 0.2 0.2, 0.1 0.1))").unwrap();
        assert!(matches!(p, Some(WktGeometry::Polygon(ref p)) if p.num_holes() == 1));
        assert!(matches!(
            parse("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), EMPTY)").unwrap(),
            Some(WktGeometry::MultiPolygon(ref v)) if v.len() == 2 && v[1].is_none()
        ));
        assert!(matches!(
            parse("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))").unwrap(),
            Some(WktGeometry::MultiLine(ref v)) if v.len() == 2
        ));
        assert_eq!(
            parse("# comment\nLINESTRING (0 0, 1 nan)")
                .unwrap_err()
                .to_string(),
            "invalid latitude NaN; must be between -90.0 and 90.0"
        );
    }

    #[test]
    fn errors() {
        let e = |s: &str| parse(s).unwrap_err();
        assert_eq!(e("CIRCLE (1 2)").to_string(), "unknown geo_shape [CIRCLE]");
        assert_eq!(
            e("POINT (1 2) x"),
            GeoError::Parse {
                message: "expected end of WKT string but found additional text: x".into(),
                offset: 1
            }
        );
        assert_eq!(
            e("POINT\n\r\n(1 a)"),
            GeoError::Parse {
                message: "invalid number found: a".into(),
                offset: 3
            }
        );
        assert_eq!(e("POINT [1 2]").to_string(), "expected word but found: '['");
        assert_eq!(
            e("POINT (1").to_string(),
            "expected number but found: END-OF-STREAM"
        );
        assert_eq!(
            parse_expected_type("POINT (1 2)", Some(ShapeType::Polygon))
                .unwrap_err()
                .to_string(),
            "Expected geometry type: [POLYGON], but found: [POINT]"
        );
        assert!(parse_expected_type("POINT (1 2)", Some(ShapeType::GeometryCollection)).is_ok());
        assert_eq!(ShapeType::Envelope.wkt_name(), "BBOX");
        assert_eq!(
            ShapeType::for_name("Envelope").unwrap(),
            ShapeType::Envelope
        );
    }
}
