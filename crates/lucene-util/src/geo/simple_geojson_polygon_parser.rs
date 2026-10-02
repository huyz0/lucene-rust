//! Port of `org.apache.lucene.geo.SimpleGeoJSONPolygonParser`: minimal
//! GeoJSON parsing that extracts a `Polygon` or `MultiPolygon`, at the top
//! level, as a `Feature`'s geometry, or as the geometry of a
//! `FeatureCollection`'s single feature.
//!
//! The parser works on UTF-16 units, as Java's `String.charAt` does, so
//! every offset in a `ParseException` (and every fragment it quotes) is
//! Java's. Its quirks are kept: a `\uXXXX` escape appends the code's
//! *decimal* value and then re-reads the four hex digits as plain
//! characters; `Integer.parseInt`'s `NumberFormatException`, an empty
//! polygon's `IndexOutOfBoundsException` and `o.getClass()` on a `null`
//! value (`NullPointerException`) escape as Java's runtime exceptions do.
//!
//! One narrowing: `Integer.parseInt` accepts any Unicode decimal digit
//! (`Character.digit`); this port accepts ASCII and fullwidth digits and
//! letters, so a `\u` escape spelled in, say, Devanagari digits is a
//! `NumberFormatException` here where Java parses it.

use super::polygon::Polygon;
use super::{java_double_string, java_parse_double, GeoError};

/// A parsed JSON value, as the Java parser keeps it (`Object`).
#[derive(Debug, Clone, PartialEq)]
enum Value {
    /// `null`: a JSON `null`, or a nested object (which Java parses and
    /// drops).
    Null,
    Bool(bool),
    Str(String),
    Num(f64),
    List(Vec<Value>),
}

impl Value {
    /// `String.valueOf(o)`.
    fn java_string(&self) -> String {
        match self {
            Value::Null => "null".into(),
            Value::Bool(b) => b.to_string(),
            Value::Str(s) => s.clone(),
            Value::Num(d) => java_double_string(*d),
            Value::List(l) => {
                let items: Vec<String> = l.iter().map(Value::java_string).collect();
                format!("[{}]", items.join(", "))
            }
        }
    }

    /// `o.getClass()` as printed, or the `NullPointerException` Java
    /// throws for `null`.
    fn java_class(&self) -> Result<&'static str, GeoError> {
        match self {
            Value::Null => Err(GeoError::NullPointer(
                "Cannot invoke \"Object.getClass()\" because \"o\" is null".into(),
            )),
            Value::Bool(_) => Ok("class java.lang.Boolean"),
            Value::Str(_) => Ok("class java.lang.String"),
            Value::Num(_) => Ok("class java.lang.Double"),
            Value::List(_) => Ok("class java.util.ArrayList"),
        }
    }
}

/// Port of `SimpleGeoJSONPolygonParser`.
#[derive(Debug, Clone)]
pub struct SimpleGeoJSONPolygonParser {
    input: Vec<u16>,
    upto: usize,
    poly_type: Option<&'static str>,
    coordinates: Option<Vec<Value>>,
}

#[inline]
fn is_json_whitespace(ch: u16) -> bool {
    ch == u16::from(b' ')
        || ch == u16::from(b'\t')
        || ch == u16::from(b'\n')
        || ch == u16::from(b'\r')
}

fn ch_str(ch: u16) -> String {
    String::from_utf16_lossy(&[ch])
}

/// `Character.digit(ch, 16)` for ASCII and fullwidth forms.
fn hex_digit(ch: u16) -> Option<i32> {
    let c = u32::from(ch);
    let v = match c {
        0x30..=0x39 => c - 0x30,
        0x41..=0x46 => c - 0x41 + 10,
        0x61..=0x66 => c - 0x61 + 10,
        0xFF10..=0xFF19 => c - 0xFF10,
        0xFF21..=0xFF26 => c - 0xFF21 + 10,
        0xFF41..=0xFF46 => c - 0xFF41 + 10,
        _ => return None,
    };
    Some(v as i32)
}

/// `Integer.parseInt(s, 16)` over four UTF-16 units.
fn parse_int_16(s: &[u16]) -> Result<i32, GeoError> {
    let nfe = || {
        GeoError::NumberFormat(format!(
            "For input string: \"{}\" under radix 16",
            String::from_utf16_lossy(s)
        ))
    };
    let (negative, digits) = match s.first() {
        Some(&c) if c == u16::from(b'-') => (true, &s[1..]),
        Some(&c) if c == u16::from(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return Err(nfe());
    }
    let mut v: i32 = 0;
    for &c in digits {
        v = v * 16 + hex_digit(c).ok_or_else(nfe)?;
    }
    Ok(if negative { -v } else { v })
}

impl SimpleGeoJSONPolygonParser {
    /// `new SimpleGeoJSONPolygonParser(input)`.
    pub fn new(input: &str) -> SimpleGeoJSONPolygonParser {
        SimpleGeoJSONPolygonParser {
            input: input.encode_utf16().collect(),
            upto: 0,
            poly_type: None,
            coordinates: None,
        }
    }

    /// `parse()`: the polygon(s), parsing the whole input.
    pub fn parse(mut self) -> Result<Vec<Polygon>, GeoError> {
        // parse entire object
        self.parse_object("")?;
        // make sure there's nothing left:
        self.read_end()?;
        let coordinates = match self.coordinates.take() {
            Some(c) => c,
            None => return Err(self.new_parse_exception("did not see any polygon coordinates")),
        };
        let poly_type = match self.poly_type {
            Some(t) => t,
            None => {
                return Err(self.new_parse_exception("did not see type: Polygon or MultiPolygon"))
            }
        };
        if poly_type == "Polygon" {
            return Ok(vec![self.parse_polygon(&coordinates)?]);
        }
        let mut polygons = Vec::new();
        for o in &coordinates {
            match o {
                Value::List(l) => polygons.push(self.parse_polygon(l)?),
                other => {
                    let class = other.java_class()?;
                    return Err(self.new_parse_exception(&format!(
                        "elements of coordinates array should be an array, but got: {class}"
                    )));
                }
            }
        }
        Ok(polygons)
    }

    /// `parseObject(path)`; `path` is the "address" by keys of where we
    /// are, e.g. `geometry.coordinates`.
    fn parse_object(&mut self, path: &str) -> Result<(), GeoError> {
        self.scan(b'{')?;
        let mut first = true;
        loop {
            let mut ch = self.peek()?;
            if ch == u16::from(b'}') {
                break;
            } else if !first {
                if ch == u16::from(b',') {
                    // ok
                    self.upto += 1;
                    ch = self.peek()?;
                    if ch == u16::from(b'}') {
                        break;
                    }
                } else {
                    return Err(
                        self.new_parse_exception(&format!("expected , but got {}", ch_str(ch)))
                    );
                }
            }
            first = false;
            let mut upto_start = self.upto;
            let key = self.parse_string()?;
            if path == "crs.properties" && key == "href" {
                self.upto = upto_start;
                return Err(self.new_parse_exception("cannot handle linked crs"));
            }
            self.scan(b':')?;
            ch = self.peek()?;
            upto_start = self.upto;
            let new_path = || {
                if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                }
            };
            let o = if ch == u16::from(b'[') {
                Value::List(self.parse_array(&new_path())?)
            } else if ch == u16::from(b'{') {
                self.parse_object(&new_path())?;
                Value::Null
            } else if ch == u16::from(b'"') {
                Value::Str(self.parse_string()?)
            } else if ch == u16::from(b't') {
                self.scan_str("true")?;
                Value::Bool(true)
            } else if ch == u16::from(b'f') {
                self.scan_str("false")?;
                Value::Bool(false)
            } else if ch == u16::from(b'n') {
                self.scan_str("null")?;
                Value::Null
            } else if ch == u16::from(b'-')
                || ch == u16::from(b'.')
                || (u16::from(b'0')..=u16::from(b'9')).contains(&ch)
            {
                Value::Num(self.parse_number()?)
            } else if ch == u16::from(b'}') {
                break;
            } else {
                return Err(self.new_parse_exception(&format!(
                    "expected array, object, string or literal value, but got: {}",
                    ch_str(ch)
                )));
            };
            if path == "crs.properties" && key == "name" {
                match &o {
                    Value::Str(crs) => {
                        if !crs.starts_with("urn:ogc:def:crs:OGC") || !crs.ends_with(":CRS84") {
                            self.upto = upto_start;
                            return Err(self.new_parse_exception(&format!(
                                "crs must be CRS84 from OGC, but saw: {}",
                                o.java_string()
                            )));
                        }
                    }
                    _ => {
                        self.upto = upto_start;
                        return Err(self.new_parse_exception(&format!(
                            "crs.properties.name should be a string, but saw: {}",
                            o.java_string()
                        )));
                    }
                }
            }
            if key == "type" && !path.starts_with("crs") {
                let ty = match &o {
                    Value::Str(t) => t.clone(),
                    _ => {
                        self.upto = upto_start;
                        return Err(self.new_parse_exception(&format!(
                            "type should be a string, but got: {}",
                            o.java_string()
                        )));
                    }
                };
                if ty == "Polygon" && is_valid_geometry_path(path) {
                    self.poly_type = Some("Polygon");
                } else if ty == "MultiPolygon" && is_valid_geometry_path(path) {
                    self.poly_type = Some("MultiPolygon");
                } else if (ty == "FeatureCollection" || ty == "Feature")
                    && (path == "features.[]" || path.is_empty())
                {
                    // OK, we recurse
                } else {
                    self.upto = upto_start;
                    return Err(self.new_parse_exception(&format!(
                        "can only handle type FeatureCollection (if it has a single polygon geometry), Feature, Polygon or MultiPolygon, but got {ty}"
                    )));
                }
            } else if key == "coordinates" && is_valid_geometry_path(path) {
                let list = match o {
                    Value::List(l) => l,
                    other => {
                        self.upto = upto_start;
                        let class = other.java_class()?;
                        return Err(self.new_parse_exception(&format!(
                            "coordinates should be an array, but got: {class}"
                        )));
                    }
                };
                if self.coordinates.is_some() {
                    self.upto = upto_start;
                    return Err(
                        self.new_parse_exception("only one Polygon or MultiPolygon is supported")
                    );
                }
                self.coordinates = Some(list);
            }
        }
        self.scan(b'}')
    }

    fn parse_polygon(&self, coordinates: &[Value]) -> Result<Polygon, GeoError> {
        let o = coordinates.first().ok_or_else(|| {
            GeoError::IndexOutOfBounds(format!(
                "Index 0 out of bounds for length {}",
                coordinates.len()
            ))
        })?;
        let poly_points = match o {
            Value::List(l) => self.parse_points(l)?,
            other => {
                return Err(self.new_parse_exception(&format!(
                    "first element of polygon array must be an array [[lat, lon], [lat, lon] ...] but got: {}",
                    other.java_string()
                )))
            }
        };
        let mut holes = Vec::new();
        for o in &coordinates[1..] {
            match o {
                Value::List(l) => {
                    let (lats, lons) = self.parse_points(l)?;
                    holes.push(Polygon::new(&lats, &lons, vec![])?);
                }
                other => {
                    return Err(self.new_parse_exception(&format!(
                        "elements of coordinates array must be an array [[lat, lon], [lat, lon] ...] but got: {}",
                        other.java_string()
                    )))
                }
            }
        }
        Polygon::new(&poly_points.0, &poly_points.1, holes)
    }

    /// `parsePoints`: `[[lon, lat], ...]` into (lats, lons).
    fn parse_points(&self, o: &[Value]) -> Result<(Vec<f64>, Vec<f64>), GeoError> {
        let mut lats = Vec::with_capacity(o.len());
        let mut lons = Vec::with_capacity(o.len());
        for point in o {
            let point_list = match point {
                Value::List(l) => l,
                other => {
                    return Err(self.new_parse_exception(&format!(
                        "elements of coordinates array must [lat, lon] array, but got: {}",
                        other.java_string()
                    )))
                }
            };
            if point_list.len() != 2 {
                return Err(self.new_parse_exception(&format!(
                    "elements of coordinates array must [lat, lon] array, but got wrong element count: {}",
                    point.java_string()
                )));
            }
            let lon = match point_list[0] {
                Value::Num(d) => d,
                ref other => {
                    return Err(self.new_parse_exception(&format!(
                        "elements of coordinates array must [lat, lon] array, but first element is not a Double: {}",
                        other.java_string()
                    )))
                }
            };
            let lat = match point_list[1] {
                Value::Num(d) => d,
                ref other => {
                    return Err(self.new_parse_exception(&format!(
                        "elements of coordinates array must [lat, lon] array, but second element is not a Double: {}",
                        other.java_string()
                    )))
                }
            };
            // lon, lat ordering in GeoJSON!
            lons.push(lon);
            lats.push(lat);
        }
        Ok((lats, lons))
    }

    fn parse_array(&mut self, path: &str) -> Result<Vec<Value>, GeoError> {
        let mut result = Vec::new();
        self.scan(b'[')?;
        while self.upto < self.input.len() {
            let mut ch = self.peek()?;
            if ch == u16::from(b']') {
                self.scan(b']')?;
                return Ok(result);
            }
            if !result.is_empty() {
                if ch != u16::from(b',') {
                    return Err(self.new_parse_exception(&format!(
                        "expected ',' separating list items, but got '{}'",
                        ch_str(ch)
                    )));
                }
                // skip the ,
                self.upto += 1;
                if self.upto == self.input.len() {
                    return Err(self.new_parse_exception("hit EOF while parsing array"));
                }
                ch = self.peek()?;
            }
            let o = if ch == u16::from(b'[') {
                Value::List(self.parse_array(&format!("{path}.[]"))?)
            } else if ch == u16::from(b'{') {
                // This is only used when parsing the "features" in type: FeatureCollection
                self.parse_object(&format!("{path}.[]"))?;
                Value::Null
            } else if ch == u16::from(b'-')
                || ch == u16::from(b'.')
                || (u16::from(b'0')..=u16::from(b'9')).contains(&ch)
            {
                Value::Num(self.parse_number()?)
            } else if ch == u16::from(b'"') {
                Value::Str(self.parse_string()?)
            } else {
                return Err(self.new_parse_exception(&format!(
                    "expected another array or number while parsing array, not '{}'",
                    ch_str(ch)
                )));
            };
            result.push(o);
        }
        Err(self.new_parse_exception("hit EOF while reading array"))
    }

    fn parse_number(&mut self) -> Result<f64, GeoError> {
        let mut b = String::new();
        let upto_start = self.upto;
        while self.upto < self.input.len() {
            let ch = self.input[self.upto];
            if ch == u16::from(b'-')
                || ch == u16::from(b'.')
                || (u16::from(b'0')..=u16::from(b'9')).contains(&ch)
                || ch == u16::from(b'e')
                || ch == u16::from(b'E')
            {
                self.upto += 1;
                b.push(ch as u8 as char);
            } else {
                break;
            }
        }
        // we only handle doubles
        match java_parse_double(&b) {
            Some(d) => Ok(d),
            None => {
                self.upto = upto_start;
                Err(self.new_parse_exception("could not parse number as double"))
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, GeoError> {
        self.scan(b'"')?;
        let mut b: Vec<u16> = Vec::new();
        while self.upto < self.input.len() {
            let mut ch = self.input[self.upto];
            if ch == u16::from(b'"') {
                self.upto += 1;
                return Ok(String::from_utf16_lossy(&b));
            }
            if ch == u16::from(b'\\') {
                // an escaped character
                self.upto += 1;
                if self.upto == self.input.len() {
                    return Err(self.new_parse_exception("hit EOF inside string literal"));
                }
                ch = self.input[self.upto];
                if ch == u16::from(b'u') {
                    // 4 hex digit unicode BMP escape
                    self.upto += 1;
                    if self.upto + 4 > self.input.len() {
                        return Err(self.new_parse_exception("hit EOF inside string literal"));
                    }
                    // Java appends the int (its decimal digits) and does not
                    // advance past the hex digits.
                    let v = parse_int_16(&self.input[self.upto..self.upto + 4])?;
                    b.extend(v.to_string().encode_utf16());
                } else if ch == u16::from(b'\\') {
                    b.push(u16::from(b'\\'));
                    self.upto += 1;
                } else {
                    return Err(self.new_parse_exception(&format!(
                        "unsupported string escape character \\{}",
                        ch_str(ch)
                    )));
                }
            } else {
                b.push(ch);
                self.upto += 1;
            }
        }
        Err(self.new_parse_exception("hit EOF inside string literal"))
    }

    fn peek(&mut self) -> Result<u16, GeoError> {
        while self.upto < self.input.len() {
            let ch = self.input[self.upto];
            if is_json_whitespace(ch) {
                self.upto += 1;
                continue;
            }
            return Ok(ch);
        }
        Err(self.new_parse_exception("unexpected EOF"))
    }

    /// `scan(char)`: skips whitespace and consumes the expected character.
    fn scan(&mut self, expected: u8) -> Result<(), GeoError> {
        while self.upto < self.input.len() {
            let ch = self.input[self.upto];
            if is_json_whitespace(ch) {
                self.upto += 1;
                continue;
            }
            if ch != u16::from(expected) {
                return Err(self.new_parse_exception(&format!(
                    "expected '{}' but got '{}'",
                    expected as char,
                    ch_str(ch)
                )));
            }
            self.upto += 1;
            return Ok(());
        }
        Err(self.new_parse_exception(&format!("expected '{}' but got EOF", expected as char)))
    }

    fn read_end(&mut self) -> Result<(), GeoError> {
        while self.upto < self.input.len() {
            let ch = self.input[self.upto];
            if !is_json_whitespace(ch) {
                return Err(self.new_parse_exception(&format!(
                    "unexpected character '{}' after end of GeoJSON object",
                    ch_str(ch)
                )));
            }
            self.upto += 1;
        }
        Ok(())
    }

    /// `scan(String)`: the expected literal.
    fn scan_str(&mut self, expected: &str) -> Result<(), GeoError> {
        let exp: Vec<u16> = expected.encode_utf16().collect();
        if self.upto + exp.len() > self.input.len() {
            return Err(self.new_parse_exception(&format!("expected \"{expected}\" but hit EOF")));
        }
        let sub = &self.input[self.upto..self.upto + exp.len()];
        if sub != exp.as_slice() {
            return Err(self.new_parse_exception(&format!(
                "expected \"{expected}\" but got \"{}\"",
                String::from_utf16_lossy(sub)
            )));
        }
        self.upto += exp.len();
        Ok(())
    }

    /// `newParseException`: the details, the offset, and up to 50 units of
    /// context.
    fn new_parse_exception(&self, details: &str) -> GeoError {
        let end = self.input.len().min(self.upto + 1);
        let fragment = if self.upto < 50 {
            String::from_utf16_lossy(&self.input[..end])
        } else {
            format!(
                "...{}",
                String::from_utf16_lossy(&self.input[self.upto - 50..end])
            )
        };
        GeoError::Parse {
            message: format!(
                "{details} at character offset {}; fragment leading to this:\n{fragment}",
                self.upto
            ),
            offset: self.upto as i32,
        }
    }
}

/// `isValidGeometryPath`: where a Multi/Polygon geometry may appear.
fn is_valid_geometry_path(path: &str) -> bool {
    path.is_empty() || path == "geometry" || path == "features.[].geometry"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Result<Vec<Polygon>, GeoError> {
        SimpleGeoJSONPolygonParser::new(s).parse()
    }

    #[test]
    fn polygons() {
        let p = parse(r#"{"type": "Polygon", "coordinates": [[[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]]}"#).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].poly_lon(1), 101.0);
        let p = parse(r#"{"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[[[0,0],[1,0],[1,1],[0,0]]], [[[5,5],[6,5],[6,6],[5,5]], [[5.1,5.1],[5.2,5.1],[5.2,5.2],[5.1,5.1]]]]}, "properties": {"a": "b\\c", "n": null, "t": true, "f": false, "x": 1e3}}"#).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[1].num_holes(), 1);
        assert_eq!(
            Polygon::from_geojson(
                r#"{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]],}"#
            )
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn errors() {
        let e = |s: &str| parse(s).unwrap_err();
        assert!(
            matches!(e(r#"{"type": "Polygon"}"#), GeoError::Parse { ref message, .. } if message.starts_with("did not see any polygon coordinates"))
        );
        assert!(e(r#"{"coordinates": [[[0,0]]]}"#)
            .to_string()
            .starts_with("did not see type: Polygon or MultiPolygon"));
        assert!(matches!(
            e(r#"{"type": "Polygon", "coordinates": []}"#),
            GeoError::IndexOutOfBounds(_)
        ));
        assert!(matches!(
            e(r#"{"type": "Polygon", "coordinates": null}"#),
            GeoError::NullPointer(_)
        ));
        assert!(e(r#"{"type": "Polygon", "coordinates": 5}"#)
            .to_string()
            .starts_with("coordinates should be an array, but got: class java.lang.Double"));
        assert!(e(r#"{"type": 5}"#)
            .to_string()
            .starts_with("type should be a string, but got: 5.0"));
        assert!(e(r#"{"type": "Point"}"#)
            .to_string()
            .starts_with("can only handle type"));
        assert!(e(r#"{"a": "\uzzzz"}"#)
            .to_string()
            .starts_with("For input string: \"zzzz\""));
        assert!(e(r#"{"a": "\q"}"#)
            .to_string()
            .starts_with("unsupported string escape"));
        assert!(e(r#"{"a": tru}"#)
            .to_string()
            .starts_with("expected \"true\" but got"));
        assert!(e(r#"{"a": nul"#)
            .to_string()
            .starts_with("expected \"null\" but hit EOF"));
        assert!(e(r#"{"a": --}"#)
            .to_string()
            .starts_with("could not parse number as double"));
        assert!(e(r#"{"a": [1 2]}"#)
            .to_string()
            .starts_with("expected ',' separating list items"));
        assert!(e(r#"{"a": [1,"#)
            .to_string()
            .starts_with("hit EOF while parsing array"));
        assert!(e(r#"{"a": [1"#)
            .to_string()
            .starts_with("hit EOF while reading array"));
        assert!(e(r#"{"a": [x]}"#)
            .to_string()
            .starts_with("expected another array or number"));
        assert!(e(r#"{"a": x}"#)
            .to_string()
            .starts_with("expected array, object, string or literal value"));
        assert!(e(r#"{"a": 1 "b": 2}"#)
            .to_string()
            .starts_with("expected , but got \""));
        assert!(e(r#"{"a": 1} x"#)
            .to_string()
            .starts_with("unexpected character 'x'"));
        assert!(e(r#"{"crs": {"properties": {"href": "x"}}}"#)
            .to_string()
            .starts_with("cannot handle linked crs"));
        assert!(e(r#"{"crs": {"properties": {"name": 1}}}"#)
            .to_string()
            .starts_with("crs.properties.name should be a string"));
        assert!(e(r#"{"crs": {"properties": {"name": "x"}}}"#)
            .to_string()
            .starts_with("crs must be CRS84"));
        assert!(e(r#"{"type": "MultiPolygon", "coordinates": [5]}"#)
            .to_string()
            .starts_with("elements of coordinates array should be an array"));
        assert!(e(r#"{"type": "Polygon", "coordinates": [5]}"#)
            .to_string()
            .starts_with("first element of polygon array"));
        assert!(
            e(r#"{"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]], 5]}"#)
                .to_string()
                .starts_with("elements of coordinates array must be an array")
        );
        assert!(e(r#"{"type": "Polygon", "coordinates": [[5]]}"#)
            .to_string()
            .starts_with("elements of coordinates array must [lat, lon] array, but got: 5.0"));
        assert!(e(r#"{"type": "Polygon", "coordinates": [[[1]]]}"#)
            .to_string()
            .contains("wrong element count: [1.0]"));
        assert!(e(r#"{"type": "Polygon", "coordinates": [[["a", 1]]]}"#)
            .to_string()
            .contains("first element is not a Double: a"));
        assert!(e(r#"{"type": "Polygon", "coordinates": [[[1, [2]]]]}"#)
            .to_string()
            .contains("second element is not a Double: [2.0]"));
        assert!(e(r#"{"type": "Polygon", "type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]]], "coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}"#).to_string().starts_with("only one Polygon"));
        assert!(e("").to_string().starts_with("expected '{' but got EOF"));
        assert!(e("[").to_string().starts_with("expected '{' but got '['"));
        assert!(e("{\"a")
            .to_string()
            .starts_with("hit EOF inside string literal"));
        assert!(e("{\"a\\")
            .to_string()
            .starts_with("hit EOF inside string literal"));
        assert!(e("{\"a\\u12")
            .to_string()
            .starts_with("hit EOF inside string literal"));
        assert!(e("{\"a\": 1,").to_string().starts_with("unexpected EOF"));
        let long = format!("{{\"{}\": x}}", "k".repeat(60));
        assert!(e(&long).to_string().contains("\n..."));
    }

    #[test]
    fn unicode_escape_quirk() {
        // `A` appends "65" and then re-reads "0041".
        let r = parse(
            r#"{"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]]], "p": {"x": "A"}}"#,
        );
        assert!(r.is_ok());
        assert_eq!(
            parse_int_16(&"-0ff".encode_utf16().collect::<Vec<_>>()).unwrap(),
            -255
        );
        assert_eq!(
            parse_int_16(&"+０Ａ".encode_utf16().collect::<Vec<_>>()).unwrap(),
            10
        );
        assert!(parse_int_16(&"-".encode_utf16().collect::<Vec<_>>()).is_err());
        assert_eq!(
            Value::Bool(true).java_class().unwrap(),
            "class java.lang.Boolean"
        );
        assert_eq!(
            Value::Str("s".into()).java_class().unwrap(),
            "class java.lang.String"
        );
        assert_eq!(
            Value::List(vec![Value::Null]).java_class().unwrap(),
            "class java.util.ArrayList"
        );
        assert_eq!(
            Value::List(vec![Value::Null, Value::Bool(false)]).java_string(),
            "[null, false]"
        );
    }
}
