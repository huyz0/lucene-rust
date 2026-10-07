//! The subset of the Java Object Serialization Stream Protocol
//! (`java.io.ObjectInputStream`) that smartcn's dictionaries are written in:
//! `coredict.mem` and `bigramdict.mem` are `ObjectOutputStream.writeObject`
//! of primitive arrays and arrays of them (`short[]`, `char[]`, `char[][][]`,
//! `int[][]`, `long[]`, `int[]`), and `AbstractDictionary`'s
//! `ObjectInputFilter` rejects every other class.
//!
//! The grammar read here: the stream header `AC ED 00 05`, then objects,
//! each `TC_NULL`, `TC_REFERENCE` (a handle, `0x7E0000` + n) or `TC_ARRAY`
//! with its class descriptor (`TC_CLASSDESC`: the array class name such as
//! `[[C`, a `serialVersionUID`, flags, no fields, `TC_ENDBLOCKDATA`, and a
//! `TC_NULL` superclass -- or a `TC_REFERENCE` to one already read), a
//! big-endian `int` length and the elements: big-endian values for a
//! primitive array, objects for an array of arrays. Handles are assigned in
//! stream order, a descriptor's after its name and UID, an array's after its
//! descriptor.
//!
//! Every count is checked against the bytes left before anything is
//! allocated, and an element whose class is not the array's component type
//! is refused (Java's `ArrayStoreException`) as soon as its class
//! descriptor is read, before its body: each nested level then has one
//! dimension fewer than its parent, so recursion is at most
//! [`MAX_DIMENSIONS`] deep. A corrupt or hostile file is an error, never a
//! panic, a stack overflow or an outsized allocation.
//!
//! Primitive arrays are held as `Arc<[T]>`, so a reader of the stream
//! shares one array between every `TC_REFERENCE` to it, as Java shares the
//! object.

use std::sync::Arc;

use crate::SmartcnError;

const STREAM_MAGIC: u16 = 0xACED;
const STREAM_VERSION: u16 = 5;
const TC_NULL: u8 = 0x70;
const TC_REFERENCE: u8 = 0x71;
const TC_CLASSDESC: u8 = 0x72;
const TC_ENDBLOCKDATA: u8 = 0x78;
const TC_ARRAY: u8 = 0x75;
const BASE_WIRE_HANDLE: u32 = 0x7E_0000;
/// The deepest array class accepted (`[[[[C`); the dictionaries use three.
const MAX_DIMENSIONS: usize = 4;

/// A primitive array's elements, or an array of arrays' element handles.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    Short(Arc<[i16]>),
    Char(Arc<[u16]>),
    Int(Arc<[i32]>),
    Long(Arc<[i64]>),
    /// `None` is a `null` element.
    Objects(Vec<Option<usize>>),
}

#[derive(Debug, Clone, PartialEq)]
enum Handle {
    ClassDesc(String),
    /// An array: its class name, and its data (`None` while its elements are
    /// still being read).
    Array(String, Option<ArrayData>),
}

/// The objects of one stream, by handle.
#[derive(Debug, Default)]
pub struct ObjectStream {
    handles: Vec<Handle>,
}

fn corrupt(msg: impl Into<String>) -> SmartcnError {
    SmartcnError::new(format!("StreamCorruptedException: {}", msg.into()))
}

/// An element of class `name` stored in an array whose component type is
/// `expected`: Java's `ArrayStoreException` when they differ.
fn check_component(name: &str, expected: Option<&str>) -> Result<(), SmartcnError> {
    match expected {
        Some(want) if want != name => Err(SmartcnError::new(format!(
            "ArrayStoreException: {name} in an array of {want}"
        ))),
        _ => Ok(()),
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], SmartcnError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| SmartcnError::new("EOFException"))?;
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, SmartcnError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SmartcnError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn i32(&mut self) -> Result<i32, SmartcnError> {
        let b = self.take(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
}

impl ObjectStream {
    /// Reads every object of `bytes`, returning the stream and the top-level
    /// objects in order (`None` for a top-level `null`).
    pub fn read(bytes: &[u8]) -> Result<(ObjectStream, Vec<Option<usize>>), SmartcnError> {
        let mut r = Reader { bytes, pos: 0 };
        if r.u16()? != STREAM_MAGIC || r.u16()? != STREAM_VERSION {
            return Err(corrupt("invalid stream header"));
        }
        let mut s = ObjectStream::default();
        let mut top = Vec::new();
        while r.remaining() > 0 {
            top.push(s.object(&mut r, None)?);
        }
        Ok((s, top))
    }

    /// One object; `expected` is the class an element must have.
    fn object(
        &mut self,
        r: &mut Reader,
        expected: Option<&str>,
    ) -> Result<Option<usize>, SmartcnError> {
        match r.u8()? {
            TC_NULL => Ok(None),
            TC_REFERENCE => {
                let h = self.reference(r)?;
                let Handle::Array(name, _) = &self.handles[h] else {
                    return Err(corrupt("reference to a class descriptor as an object"));
                };
                check_component(name, expected)?;
                Ok(Some(h))
            }
            TC_ARRAY => self.array(r, expected).map(Some),
            tc => Err(corrupt(format!("invalid type code: {tc:02X}"))),
        }
    }

    fn reference(&mut self, r: &mut Reader) -> Result<usize, SmartcnError> {
        let wire = r.i32()? as u32;
        wire.checked_sub(BASE_WIRE_HANDLE)
            .map(|h| h as usize)
            .filter(|&h| h < self.handles.len())
            .ok_or_else(|| corrupt(format!("invalid handle value: {wire:08X}")))
    }

    /// A class descriptor (`TC_CLASSDESC` or a reference to one): its name.
    fn class_desc(&mut self, r: &mut Reader) -> Result<String, SmartcnError> {
        match r.u8()? {
            TC_CLASSDESC => {
                let len = usize::from(r.u16()?);
                let name = std::str::from_utf8(r.take(len)?)
                    .map_err(|_| corrupt("class name is not UTF-8"))?
                    .to_string();
                r.take(8)?; // serialVersionUID
                self.handles.push(Handle::ClassDesc(name.clone()));
                r.u8()?; // flags
                if r.u16()? != 0 {
                    return Err(corrupt(format!("{name}: an array class has no fields")));
                }
                if r.u8()? != TC_ENDBLOCKDATA {
                    return Err(corrupt(format!("{name}: class annotations")));
                }
                if r.u8()? != TC_NULL {
                    return Err(corrupt(format!("{name}: an array class has no superclass")));
                }
                Ok(name)
            }
            TC_REFERENCE => {
                let h = self.reference(r)?;
                match &self.handles[h] {
                    Handle::ClassDesc(name) => Ok(name.clone()),
                    Handle::Array(..) => Err(corrupt("reference to an array as a class")),
                }
            }
            tc => Err(corrupt(format!("invalid class descriptor code: {tc:02X}"))),
        }
    }

    /// An array, after its `TC_ARRAY`; `expected` as for [`Self::object`],
    /// checked before the body is read, so each nested level has one
    /// dimension fewer than its parent and recursion is at most
    /// [`MAX_DIMENSIONS`] deep.
    fn array(&mut self, r: &mut Reader, expected: Option<&str>) -> Result<usize, SmartcnError> {
        let name = self.class_desc(r)?;
        check_component(&name, expected)?;
        let handle = self.handles.len();
        self.handles.push(Handle::Array(name.clone(), None));
        let len = r.i32()?;
        let len = usize::try_from(len)
            .map_err(|_| SmartcnError::new(format!("NegativeArraySizeException: {len}")))?;
        let dims = name.bytes().take_while(|&b| b == b'[').count();
        let component = name.get(1..).unwrap_or("");
        let fits = |width: usize| len.checked_mul(width).is_some_and(|n| n <= r.remaining());
        let data = match component {
            "S" | "C" | "I" | "J" => {
                let width = match component {
                    "S" | "C" => 2,
                    "I" => 4,
                    _ => 8,
                };
                if !fits(width) {
                    return Err(SmartcnError::new("EOFException"));
                }
                let raw = r.take(len.saturating_mul(width))?;
                match component {
                    "S" => ArrayData::Short(
                        raw.chunks_exact(2)
                            .map(|c| i16::from_be_bytes([c[0], c[1]]))
                            .collect(),
                    ),
                    "C" => ArrayData::Char(
                        raw.chunks_exact(2)
                            .map(|c| u16::from_be_bytes([c[0], c[1]]))
                            .collect(),
                    ),
                    "I" => ArrayData::Int(
                        raw.chunks_exact(4)
                            .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                    ),
                    _ => ArrayData::Long(
                        raw.chunks_exact(8)
                            .map(|c| {
                                i64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect(),
                    ),
                }
            }
            c if c.starts_with('[') && (2..=MAX_DIMENSIONS).contains(&dims) => {
                // Every element is at least one byte (TC_NULL).
                if !fits(1) {
                    return Err(SmartcnError::new("EOFException"));
                }
                let mut elements = Vec::with_capacity(len);
                for _ in 0..len {
                    elements.push(self.object(r, Some(c))?);
                }
                ArrayData::Objects(elements)
            }
            _ => {
                return Err(SmartcnError::new(format!(
                    "InvalidClassException: filter status: REJECTED ({name})"
                )))
            }
        };
        self.handles[handle] = Handle::Array(name, Some(data));
        Ok(handle)
    }

    /// The array behind `handle`: its class name and data; `None` for an
    /// array still being read (a reference into itself) or a descriptor.
    pub fn array_at(&self, handle: usize) -> Option<(&str, &ArrayData)> {
        match self.handles.get(handle)? {
            Handle::Array(name, Some(data)) => Some((name, data)),
            _ => None,
        }
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    //! A test-only writer of the same subset, as `ObjectOutputStream` lays
    //! it out (descriptors shared by reference after their first use).
    #![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

    use std::collections::HashMap;

    pub(crate) enum Value {
        Null,
        Short(Vec<i16>),
        Char(Vec<u16>),
        Int(Vec<i32>),
        Long(Vec<i64>),
        Objects(&'static str, Vec<Value>),
    }

    #[derive(Default)]
    pub(crate) struct Writer {
        pub(crate) out: Vec<u8>,
        classes: HashMap<String, u32>,
        next: u32,
    }

    impl Writer {
        pub(crate) fn new() -> Self {
            let mut w = Writer::default();
            w.out.extend_from_slice(&[0xAC, 0xED, 0x00, 0x05]);
            w
        }

        fn desc(&mut self, name: &str) {
            if let Some(&h) = self.classes.get(name) {
                self.out.push(0x71);
                self.out.extend_from_slice(&(0x7E_0000 + h).to_be_bytes());
                return;
            }
            self.out.push(0x72);
            self.out
                .extend_from_slice(&(name.len() as u16).to_be_bytes());
            self.out.extend_from_slice(name.as_bytes());
            self.out.extend_from_slice(&[0; 8]);
            self.classes.insert(name.to_string(), self.next);
            self.next += 1;
            self.out.extend_from_slice(&[0x02, 0x00, 0x00, 0x78, 0x70]);
        }

        pub(crate) fn write(&mut self, v: &Value) {
            let (name, len) = match v {
                Value::Null => {
                    self.out.push(0x70);
                    return;
                }
                Value::Short(x) => ("[S", x.len()),
                Value::Char(x) => ("[C", x.len()),
                Value::Int(x) => ("[I", x.len()),
                Value::Long(x) => ("[J", x.len()),
                Value::Objects(n, x) => (*n, x.len()),
            };
            self.out.push(0x75);
            self.desc(name);
            self.next += 1;
            self.out.extend_from_slice(&(len as i32).to_be_bytes());
            match v {
                Value::Null => unreachable!(),
                Value::Short(x) => x
                    .iter()
                    .for_each(|e| self.out.extend_from_slice(&e.to_be_bytes())),
                Value::Char(x) => x
                    .iter()
                    .for_each(|e| self.out.extend_from_slice(&e.to_be_bytes())),
                Value::Int(x) => x
                    .iter()
                    .for_each(|e| self.out.extend_from_slice(&e.to_be_bytes())),
                Value::Long(x) => x
                    .iter()
                    .for_each(|e| self.out.extend_from_slice(&e.to_be_bytes())),
                Value::Objects(_, x) => x.iter().for_each(|e| self.write(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

    use super::test_util::{Value, Writer};
    use super::*;

    fn read(w: &Writer) -> Result<(ObjectStream, Vec<Option<usize>>), SmartcnError> {
        ObjectStream::read(&w.out)
    }

    #[test]
    fn primitive_and_nested_arrays_round_trip() {
        let mut w = Writer::new();
        w.write(&Value::Short(vec![-1, 2]));
        w.write(&Value::Char(vec![0x4E2D]));
        w.write(&Value::Objects(
            "[[[C",
            vec![
                Value::Null,
                Value::Objects("[[C", vec![Value::Char(vec![1, 2]), Value::Null]),
                Value::Objects("[[C", vec![]),
            ],
        ));
        w.write(&Value::Objects(
            "[[I",
            vec![Value::Int(vec![7]), Value::Null],
        ));
        w.write(&Value::Long(vec![i64::MIN]));
        w.write(&Value::Null);
        let (s, top) = read(&w).unwrap();
        assert_eq!(top.len(), 6);
        assert_eq!(top[5], None);
        assert_eq!(
            s.array_at(top[0].unwrap()),
            Some(("[S", &ArrayData::Short(vec![-1, 2].into())))
        );
        assert_eq!(
            s.array_at(top[1].unwrap()),
            Some(("[C", &ArrayData::Char(vec![0x4E2D].into())))
        );
        let Some((name, ArrayData::Objects(outer))) = s.array_at(top[2].unwrap()) else {
            panic!()
        };
        assert_eq!((name, outer.len(), outer[0]), ("[[[C", 3, None));
        let Some(("[[C", ArrayData::Objects(inner))) = s.array_at(outer[1].unwrap()) else {
            panic!()
        };
        assert_eq!(
            s.array_at(inner[0].unwrap()),
            Some(("[C", &ArrayData::Char(vec![1, 2].into())))
        );
        assert_eq!(
            s.array_at(top[4].unwrap()),
            Some(("[J", &ArrayData::Long(vec![i64::MIN].into())))
        );
        assert_eq!(s.array_at(0), None); // a descriptor
        assert_eq!(s.array_at(10_000), None);
    }

    #[test]
    fn references_to_earlier_arrays() {
        let mut w = Writer::new();
        w.write(&Value::Char(vec![9]));
        // [[C of two elements: the first array by reference, then a null.
        w.out
            .extend_from_slice(&[0x75, 0x72, 0, 3, b'[', b'[', b'C', 0, 0, 0, 0, 0, 0, 0, 0]);
        w.out
            .extend_from_slice(&[2, 0, 0, 0x78, 0x70, 0, 0, 0, 2, 0x71, 0, 0x7E, 0, 1, 0x70]);
        let (s, top) = read(&w).unwrap();
        let Some((_, ArrayData::Objects(e))) = s.array_at(top[1].unwrap()) else {
            panic!()
        };
        assert_eq!(e, &vec![top[0], None]);
    }

    #[test]
    fn malformed_streams_are_errors() {
        let msg = |b: &[u8]| match ObjectStream::read(b) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("accepted {b:02X?}"),
        };
        assert!(msg(&[0xAC, 0xED, 0, 4]).contains("invalid stream header"));
        assert!(msg(&[0xAC]).contains("EOF"));
        assert!(msg(&[0xAC, 0xED, 0, 5, 0x42]).contains("invalid type code: 42"));
        // A reference before any handle; to a descriptor as an object.
        assert!(msg(&[0xAC, 0xED, 0, 5, 0x71, 0, 0x7E, 0, 0]).contains("invalid handle"));
        let mut w = Writer::new();
        w.write(&Value::Int(vec![1]));
        let mut b = w.out.clone();
        b.extend_from_slice(&[0x71, 0, 0x7E, 0, 0]);
        assert!(msg(&b).contains("class descriptor as an object"));
        // An array descriptor referencing an array handle.
        let mut b = w.out.clone();
        b.extend_from_slice(&[0x75, 0x71, 0, 0x7E, 0, 1]);
        assert!(msg(&b).contains("array as a class"));
        assert!(msg(&[0xAC, 0xED, 0, 5, 0x75, 0x42]).contains("descriptor code: 42"));
        // A class that is not a primitive array (the filter's REJECTED).
        let mut w = Writer::new();
        w.write(&Value::Objects("[Ljava.lang.String;", vec![]));
        assert!(msg(&w.out).contains("REJECTED"));
        // A component of the wrong class.
        let mut w = Writer::new();
        w.write(&Value::Objects("[[C", vec![Value::Int(vec![1])]));
        assert!(msg(&w.out).contains("ArrayStoreException"));
        // Lengths: negative, past the end, past the end for objects.
        let mut w = Writer::new();
        w.write(&Value::Int(vec![1, 2]));
        let mut b = w.out.clone();
        let n = b.len();
        b[n - 12..n - 8].copy_from_slice(&(-1i32).to_be_bytes());
        assert!(msg(&b).contains("NegativeArraySize"));
        b[n - 12..n - 8].copy_from_slice(&i32::MAX.to_be_bytes());
        assert!(msg(&b).contains("EOF"));
        let mut w = Writer::new();
        w.write(&Value::Objects("[[C", vec![Value::Null]));
        let mut b = w.out.clone();
        let n = b.len();
        b[n - 5..n - 1].copy_from_slice(&1000i32.to_be_bytes());
        assert!(msg(&b).contains("EOF"));
        // Descriptor shapes Java's array classes never have.
        for (i, (byte, want)) in [
            (1u8, "no fields"),
            (0x77, "annotations"),
            (0x72, "no superclass"),
        ]
        .into_iter()
        .enumerate()
        {
            let mut w = Writer::new();
            w.write(&Value::Int(vec![]));
            let pos = [20, 21, 22][i];
            w.out[pos] = byte;
            assert!(msg(&w.out).contains(want), "{want}: {}", msg(&w.out));
        }
        let mut w = Writer::new();
        w.write(&Value::Int(vec![]));
        w.out[8] = 0xFF;
        assert!(msg(&w.out).contains("UTF-8"));
        // Too deep.
        let mut w = Writer::new();
        w.write(&Value::Objects("[[[[[C", vec![]));
        assert!(msg(&w.out).contains("REJECTED"));
    }

    #[test]
    fn a_self_reference_is_refused() {
        // [[C whose element refers to itself: the wrong component class, so
        // an array can never contain itself.
        let mut b = vec![0xAC, 0xED, 0, 5, 0x75, 0x72, 0, 3, b'[', b'[', b'C'];
        b.extend_from_slice(&[0; 8]);
        b.extend_from_slice(&[2, 0, 0, 0x78, 0x70, 0, 0, 0, 1, 0x71, 0, 0x7E, 0, 1]);
        let e = ObjectStream::read(&b).err().unwrap();
        assert!(e.to_string().contains("ArrayStoreException"), "{e}");
    }
}
