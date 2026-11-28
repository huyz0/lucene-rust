//! A statement's result as JSON, in the shape of the SQL plugin's
//! `_plugins/_sql` response: a `schema` of `name` and `type`, the `datarows`,
//! and the row counts.

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, FixedSizeListArray, Float32Array, Float64Array,
    RecordBatch, StringArray,
};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, Schema,
    UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};

use crate::error::Result;

/// One column of a batch, read without a per-value type test.
enum Col<'a> {
    Bool(&'a BooleanArray),
    I8(&'a datafusion::arrow::array::Int8Array),
    I16(&'a datafusion::arrow::array::Int16Array),
    I32(&'a datafusion::arrow::array::Int32Array),
    I64(&'a datafusion::arrow::array::Int64Array),
    U8(&'a datafusion::arrow::array::UInt8Array),
    U16(&'a datafusion::arrow::array::UInt16Array),
    U32(&'a datafusion::arrow::array::UInt32Array),
    U64(&'a datafusion::arrow::array::UInt64Array),
    F32(&'a Float32Array),
    F64(&'a Float64Array),
    Str(&'a StringArray),
    Vector(&'a FixedSizeListArray, &'a Float32Array),
    Null,
}

/// The type name a column has in the response's `schema`.
fn type_name(t: &DataType) -> &'static str {
    match t {
        DataType::Boolean => "boolean",
        DataType::Int8 | DataType::UInt8 => "byte",
        DataType::Int16 | DataType::UInt16 => "short",
        DataType::Int32 | DataType::UInt32 => "integer",
        DataType::Int64 | DataType::UInt64 => "long",
        DataType::Float32 => "float",
        DataType::Float64 => "double",
        DataType::FixedSizeList(..) => "knn_vector",
        DataType::Null => "undefined",
        _ => "keyword",
    }
}

/// Whether [`Col`] reads arrays of `t` as they are.
fn direct(t: &DataType) -> bool {
    match t {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::Null => true,
        DataType::FixedSizeList(f, _) => f.data_type() == &DataType::Float32,
        _ => false,
    }
}

fn column(a: &ArrayRef) -> Col<'_> {
    match a.data_type() {
        DataType::Boolean => Col::Bool(a.as_boolean()),
        DataType::Int8 => Col::I8(a.as_primitive::<Int8Type>()),
        DataType::Int16 => Col::I16(a.as_primitive::<Int16Type>()),
        DataType::Int32 => Col::I32(a.as_primitive::<Int32Type>()),
        DataType::Int64 => Col::I64(a.as_primitive::<Int64Type>()),
        DataType::UInt8 => Col::U8(a.as_primitive::<UInt8Type>()),
        DataType::UInt16 => Col::U16(a.as_primitive::<UInt16Type>()),
        DataType::UInt32 => Col::U32(a.as_primitive::<UInt32Type>()),
        DataType::UInt64 => Col::U64(a.as_primitive::<UInt64Type>()),
        DataType::Float32 => Col::F32(a.as_primitive::<Float32Type>()),
        DataType::Float64 => Col::F64(a.as_primitive::<Float64Type>()),
        DataType::Utf8 => Col::Str(a.as_string::<i32>()),
        DataType::FixedSizeList(..) => {
            let list = a.as_fixed_size_list();
            Col::Vector(list, list.values().as_primitive::<Float32Type>())
        }
        _ => Col::Null,
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Bytes that need an escape in a JSON string: control characters, `"` and `\`.
const fn escape_table() -> [bool; 256] {
    let mut t = [false; 256];
    let mut i = 0;
    while i < 0x20 {
        t[i] = true;
        i += 1;
    }
    t[b'"' as usize] = true;
    t[b'\\' as usize] = true;
    t
}

static ESCAPE: [bool; 256] = escape_table();

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    let bytes = s.as_bytes();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if !ESCAPE[usize::from(b)] {
            continue;
        }
        out.extend_from_slice(&bytes[start..i]);
        start = i + 1;
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            _ => out.extend_from_slice(&[
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX[usize::from(b >> 4)],
                HEX[usize::from(b & 15)],
            ]),
        }
    }
    out.extend_from_slice(&bytes[start..]);
    out.push(b'"');
}

fn put_f32(out: &mut Vec<u8>, v: f32) {
    if v.is_finite() {
        out.extend_from_slice(ryu::Buffer::new().format_finite(v).as_bytes());
    } else {
        out.extend_from_slice(b"null");
    }
}

fn put_f64(out: &mut Vec<u8>, v: f64) {
    if v.is_finite() {
        out.extend_from_slice(ryu::Buffer::new().format_finite(v).as_bytes());
    } else {
        out.extend_from_slice(b"null");
    }
}

fn put_int<I: itoa::Integer>(out: &mut Vec<u8>, v: I) {
    out.extend_from_slice(itoa::Buffer::new().format(v).as_bytes());
}

macro_rules! prim {
    ($a:expr, $row:expr, $out:expr, $put:expr) => {
        if $a.is_null($row) {
            $out.extend_from_slice(b"null");
        } else {
            $put($out, $a.value($row));
        }
    };
}

fn put_value(out: &mut Vec<u8>, c: &Col<'_>, row: usize) {
    match c {
        Col::Bool(a) => prim!(a, row, out, |o: &mut Vec<u8>, v: bool| o
            .extend_from_slice(if v { b"true" } else { b"false" })),
        Col::I8(a) => prim!(a, row, out, put_int),
        Col::I16(a) => prim!(a, row, out, put_int),
        Col::I32(a) => prim!(a, row, out, put_int),
        Col::I64(a) => prim!(a, row, out, put_int),
        Col::U8(a) => prim!(a, row, out, put_int),
        Col::U16(a) => prim!(a, row, out, put_int),
        Col::U32(a) => prim!(a, row, out, put_int),
        Col::U64(a) => prim!(a, row, out, put_int),
        Col::F32(a) => prim!(a, row, out, put_f32),
        Col::F64(a) => prim!(a, row, out, put_f64),
        Col::Str(a) => prim!(a, row, out, put_str),
        Col::Vector(list, values) => {
            if list.is_null(row) {
                out.extend_from_slice(b"null");
                return;
            }
            let width = usize::try_from(list.value_length()).unwrap_or(0);
            out.push(b'[');
            for i in 0..width {
                if i > 0 {
                    out.push(b',');
                }
                let at = row * width + i;
                if values.is_null(at) {
                    out.extend_from_slice(b"null");
                } else {
                    put_f32(out, values.value(at));
                }
            }
            out.push(b']');
        }
        Col::Null => out.extend_from_slice(b"null"),
    }
}

/// `batches` as the SQL plugin's JSON response, appended to `out`.
///
/// A column of a type the writer has no form for is written as its text.
///
/// # Errors
/// A column that cannot be turned into text.
pub fn write_json(out: &mut Vec<u8>, schema: &Schema, batches: &[RecordBatch]) -> Result<()> {
    out.extend_from_slice(b"{\"schema\":[");
    for (i, f) in schema.fields().iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(b"{\"name\":");
        put_str(out, f.name());
        out.extend_from_slice(b",\"type\":\"");
        out.extend_from_slice(type_name(f.data_type()).as_bytes());
        out.extend_from_slice(b"\"}");
    }
    out.extend_from_slice(b"],\"datarows\":[");
    let mut rows = 0usize;
    for batch in batches {
        let arrays: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|a| {
                if direct(a.data_type()) {
                    Ok(Arc::clone(a))
                } else {
                    cast(a, &DataType::Utf8)
                }
            })
            .collect::<std::result::Result<_, _>>()?;
        let cols: Vec<Col<'_>> = arrays.iter().map(column).collect();
        for row in 0..batch.num_rows() {
            if rows > 0 {
                out.push(b',');
            }
            out.push(b'[');
            for (i, c) in cols.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                put_value(out, c, row);
            }
            out.push(b']');
            rows += 1;
        }
    }
    out.extend_from_slice(b"],\"total\":");
    put_int(out, rows);
    out.extend_from_slice(b",\"size\":");
    put_int(out, rows);
    out.extend_from_slice(b",\"status\":200}");
    Ok(())
}

/// The schema a result's batches share, or `fallback` for a result with no
/// batch.
pub fn result_schema(batches: &[RecordBatch], fallback: &Schema) -> Arc<Schema> {
    batches
        .first()
        .map_or_else(|| Arc::new(fallback.clone()), RecordBatch::schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int64Array, UInt64Array};
    use datafusion::arrow::datatypes::Field;

    fn json(batch: &RecordBatch) -> serde_json::Value {
        let mut out = Vec::new();
        write_json(&mut out, &batch.schema(), std::slice::from_ref(batch)).unwrap();
        serde_json::from_slice(&out).unwrap()
    }

    #[test]
    fn rows_come_out_as_the_sql_plugin_writes_them() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, true),
            Field::new("n", DataType::Int64, true),
            Field::new("u", DataType::UInt64, true),
            Field::new("x", DataType::Float64, true),
            Field::new("f", DataType::Float32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![
                    Some("a\"b\\c\n\u{1}é"),
                    None,
                ])),
                Arc::new(Int64Array::from(vec![Some(-7), None])),
                Arc::new(UInt64Array::from(vec![Some(u64::MAX), Some(0)])),
                Arc::new(Float64Array::from(vec![Some(1.5), Some(f64::NAN)])),
                Arc::new(Float32Array::from(vec![Some(0.25), Some(f32::INFINITY)])),
            ],
        )
        .unwrap();
        let v = json(&batch);
        assert_eq!(v["total"], 2);
        assert_eq!(v["size"], 2);
        assert_eq!(v["status"], 200);
        assert_eq!(v["schema"][0]["name"], "k");
        assert_eq!(v["schema"][1]["type"], "long");
        assert_eq!(v["schema"][3]["type"], "double");
        assert_eq!(
            v["datarows"][0],
            serde_json::json!(["a\"b\\c\n\u{1}é", -7, u64::MAX, 1.5, 0.25])
        );
        assert_eq!(
            v["datarows"][1],
            serde_json::json!([null, null, 0, null, null])
        );
    }

    #[test]
    fn vectors_and_other_types_are_written() {
        use datafusion::arrow::array::Date32Array;
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let list = FixedSizeListArray::new(
            item.clone(),
            2,
            Arc::new(Float32Array::from(vec![1.0, 2.5, 3.0, 4.0])),
            None,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::FixedSizeList(item, 2), true),
            Field::new("d", DataType::Date32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(list), Arc::new(Date32Array::from(vec![0, 1]))],
        )
        .unwrap();
        let v = json(&batch);
        assert_eq!(v["schema"][0]["type"], "knn_vector");
        assert_eq!(v["datarows"][0], serde_json::json!([[1.0, 2.5], "1970-01-01"]));
        assert_eq!(v["datarows"][1][0], serde_json::json!([3.0, 4.0]));
    }

    #[test]
    fn an_empty_result_still_has_its_schema() {
        let schema = Schema::new(vec![Field::new("k", DataType::Utf8, true)]);
        let mut out = Vec::new();
        write_json(&mut out, &schema, &[]).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["datarows"], serde_json::json!([]));
        assert_eq!(v["total"], 0);
    }
}
