//! The constant sources: `ConstNumberSource` with `ConstValueSource` and
//! `DoubleConstValueSource`, `LiteralValueSource`, and the constant vectors
//! `ConstKnnFloatValueSource`/`ConstKnnByteVectorValueSource`.

use crate::function::docvalues::{
    Double, DoubleDocValues, Float, FloatDocValues, Str, StrDocValues,
};
use crate::function::{
    java_byte_array, java_double, java_float, java_float_array, BoxValues, FunctionContext,
    FunctionValues, ObjectVal, ValueLeaf, ValueSource,
};
use crate::{Error, Result};

/// `ConstNumberSource`: a source with one value for every document.
pub trait ConstNumberSource: ValueSource {
    fn get_int(&self) -> i32;
    fn get_long(&self) -> i64;
    fn get_float(&self) -> f32;
    fn get_double(&self) -> f64;
    /// `getNumber()`, boxed as Java boxes it.
    fn get_number(&self) -> ObjectVal;
    fn get_bool(&self) -> bool;
}

/// `ConstValueSource`: a `float` constant.
#[derive(Debug, Clone, Copy)]
pub struct ConstValueSource {
    constant: f32,
    dv: f64,
}

impl ConstValueSource {
    pub fn new(constant: f32) -> Self {
        Self {
            constant,
            dv: f64::from(constant),
        }
    }
}

struct ConstValues {
    constant: f32,
    dv: f64,
    description: String,
}

impl FloatDocValues for ConstValues {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn float_val(&mut self, _doc: i32) -> Result<f32> {
        Ok(self.constant)
    }
    fn int_val(&mut self, _doc: i32) -> Result<i32> {
        Ok(self.constant as i32)
    }
    fn long_val(&mut self, _doc: i32) -> Result<i64> {
        Ok(self.constant as i64)
    }
    fn double_val(&mut self, _doc: i32) -> Result<f64> {
        Ok(self.dv)
    }
    fn to_string_doc(&mut self, _doc: i32) -> Result<String> {
        Ok(self.description.clone())
    }
    fn object_val(&mut self, _doc: i32) -> Result<ObjectVal> {
        Ok(ObjectVal::Float(self.constant))
    }
    fn bool_val(&mut self, _doc: i32) -> Result<bool> {
        Ok(self.constant != 0.0)
    }
}

impl ValueSource for ConstValueSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Float(ConstValues {
            constant: self.constant,
            dv: self.dv,
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        format!("const({})", java_float(self.constant))
    }
}

impl ConstNumberSource for ConstValueSource {
    fn get_int(&self) -> i32 {
        self.constant as i32
    }
    fn get_long(&self) -> i64 {
        self.constant as i64
    }
    fn get_float(&self) -> f32 {
        self.constant
    }
    fn get_double(&self) -> f64 {
        self.dv
    }
    fn get_number(&self) -> ObjectVal {
        ObjectVal::Float(self.constant)
    }
    fn get_bool(&self) -> bool {
        self.constant != 0.0
    }
}

/// `DoubleConstValueSource`: a `double` constant.
#[derive(Debug, Clone, Copy)]
pub struct DoubleConstValueSource {
    constant: f64,
    fv: f32,
    lv: i64,
}

impl DoubleConstValueSource {
    pub fn new(constant: f64) -> Self {
        Self {
            constant,
            fv: constant as f32,
            lv: constant as i64,
        }
    }
}

struct DoubleConstValues {
    constant: f64,
    fv: f32,
    lv: i64,
    description: String,
}

impl DoubleDocValues for DoubleConstValues {
    fn description(&self) -> String {
        self.description.clone()
    }
    fn double_val(&mut self, _doc: i32) -> Result<f64> {
        Ok(self.constant)
    }
    fn float_val(&mut self, _doc: i32) -> Result<f32> {
        Ok(self.fv)
    }
    /// `(int) lv`: the long truncated.
    fn int_val(&mut self, _doc: i32) -> Result<i32> {
        Ok(self.lv as i32)
    }
    fn long_val(&mut self, _doc: i32) -> Result<i64> {
        Ok(self.lv)
    }
    fn str_val(&mut self, _doc: i32) -> Result<Option<String>> {
        Ok(Some(java_double(self.constant)))
    }
    fn object_val(&mut self, _doc: i32) -> Result<ObjectVal> {
        Ok(ObjectVal::Double(self.constant))
    }
    fn to_string_doc(&mut self, _doc: i32) -> Result<String> {
        Ok(self.description.clone())
    }
}

impl ValueSource for DoubleConstValueSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Double(DoubleConstValues {
            constant: self.constant,
            fv: self.fv,
            lv: self.lv,
            description: self.description(),
        })))
    }
    fn description(&self) -> String {
        format!("const({})", java_double(self.constant))
    }
}

impl ConstNumberSource for DoubleConstValueSource {
    fn get_int(&self) -> i32 {
        self.lv as i32
    }
    fn get_long(&self) -> i64 {
        self.lv
    }
    fn get_float(&self) -> f32 {
        self.fv
    }
    fn get_double(&self) -> f64 {
        self.constant
    }
    fn get_number(&self) -> ObjectVal {
        ObjectVal::Double(self.constant)
    }
    fn get_bool(&self) -> bool {
        self.constant != 0.0
    }
}

/// `LiteralValueSource`: a string constant.
#[derive(Debug, Clone)]
pub struct LiteralValueSource {
    string: String,
}

impl LiteralValueSource {
    pub fn new(string: impl Into<String>) -> Self {
        Self {
            string: string.into(),
        }
    }

    /// `getValue()`.
    pub fn value(&self) -> &str {
        &self.string
    }
}

struct LiteralValues {
    string: String,
}

impl StrDocValues for LiteralValues {
    fn str_val(&mut self, _doc: i32) -> Result<Option<String>> {
        Ok(Some(self.string.clone()))
    }
    fn bytes_val(&mut self, _doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        target.clear();
        target.extend_from_slice(self.string.as_bytes());
        Ok(true)
    }
    fn to_string_doc(&mut self, _doc: i32) -> Result<String> {
        Ok(self.string.clone())
    }
}

impl ValueSource for LiteralValueSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(Str(LiteralValues {
            string: self.string.clone(),
        })))
    }
    fn description(&self) -> String {
        format!("literal({})", self.string)
    }
}

/// `ConstKnnFloatValueSource`: one float vector for every document.
#[derive(Debug, Clone)]
pub struct ConstKnnFloatValueSource {
    vector: Vec<f32>,
}

impl ConstKnnFloatValueSource {
    /// `new ConstKnnFloatValueSource(vector)`: `VectorUtil.checkFinite`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a non-finite component.
    pub fn new(vector: Vec<f32>) -> Result<Self> {
        if let Some((i, v)) = vector.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(Error::IllegalArgument(format!(
                "non-finite value at vector[{i}]={}",
                java_float(*v)
            )));
        }
        Ok(Self { vector })
    }
}

struct ConstFloatVectorValues {
    vector: Vec<f32>,
    description: String,
}

impl FunctionValues for ConstFloatVectorValues {
    fn float_vector_val(&mut self, _doc: i32) -> Result<Option<Vec<f32>>> {
        Ok(Some(self.vector.clone()))
    }
    fn str_val(&mut self, _doc: i32) -> Result<Option<String>> {
        Ok(Some(java_float_array(&self.vector)))
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = super::super::docvalues::opt_str(self.str_val(doc)?);
        Ok(format!("{}={s}", self.description))
    }
}

impl ValueSource for ConstKnnFloatValueSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(ConstFloatVectorValues {
            vector: self.vector.clone(),
            description: self.description(),
        }))
    }
    fn description(&self) -> String {
        format!(
            "ConstKnnFloatValueSource({})",
            java_float_array(&self.vector)
        )
    }
}

/// `ConstKnnByteVectorValueSource`: one byte vector for every document.
#[derive(Debug, Clone)]
pub struct ConstKnnByteVectorValueSource {
    vector: Vec<u8>,
}

impl ConstKnnByteVectorValueSource {
    pub fn new(vector: Vec<u8>) -> Self {
        Self { vector }
    }
}

struct ConstByteVectorValues {
    vector: Vec<u8>,
    description: String,
}

impl FunctionValues for ConstByteVectorValues {
    fn byte_vector_val(&mut self, _doc: i32) -> Result<Option<Vec<u8>>> {
        Ok(Some(self.vector.clone()))
    }
    fn str_val(&mut self, _doc: i32) -> Result<Option<String>> {
        Ok(Some(java_byte_array(&self.vector)))
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = super::super::docvalues::opt_str(self.str_val(doc)?);
        Ok(format!("{}={s}", self.description))
    }
}

impl ValueSource for ConstKnnByteVectorValueSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        _leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        Ok(Box::new(ConstByteVectorValues {
            vector: self.vector.clone(),
            description: self.description(),
        }))
    }
    fn description(&self) -> String {
        format!(
            "ConstKnnByteVectorValueSource({})",
            java_byte_array(&self.vector)
        )
    }
}
