//! `LazyDocument`: stored field values that are loaded when asked for, not
//! when the field is handed out.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue, StoredField};
use lucene_index::document::Number;

use crate::directory_reader::DirectoryReader;
use crate::{Error, Result};

/// A lazy field's slot: Java's `volatile IndexableField realValue`.
type Slot = Arc<Mutex<Option<StoredField>>>;

/// What `LazyDocument` shares between its fields.
#[derive(Default)]
struct State {
    /// The document as last loaded (`doc`), for the field names asked for so
    /// far; `None` until a value is wanted, and again after a field is added.
    doc: Option<Vec<StoredField>>,
    /// `fields`: each field number's lazy values, in the order they were handed out.
    fields: HashMap<i32, Vec<Slot>>,
    /// `fieldNames`.
    field_names: BTreeSet<String>,
}

/// `LazyDocument`: defers loading a field's value until it is asked for. The
/// fields it hands out borrow it, so they cannot outlive the reader.
pub struct LazyDocument<'r> {
    reader: &'r DirectoryReader,
    doc_id: i32,
    state: Mutex<State>,
}

impl<'r> LazyDocument<'r> {
    /// `new LazyDocument(reader, docID)`.
    pub fn new(reader: &'r DirectoryReader, doc_id: i32) -> Self {
        Self {
            reader,
            doc_id,
            state: Mutex::new(State::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panic while holding the lock leaves nothing half-written.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `getField(fieldInfo)`: a field whose value is loaded if and when it is
    /// used. Call it once for each value of the field, in order. Asking for
    /// another field after others were loaded makes the next load fetch the
    /// document again.
    pub fn get_field(&self, field_info: &FieldInfo) -> LazyField<'_, 'r> {
        let slot: Slot = Arc::new(Mutex::new(None));
        let mut state = self.lock();
        state.field_names.insert(field_info.name.clone());
        state
            .fields
            .entry(field_info.number)
            .or_default()
            .push(Arc::clone(&slot));
        // Edge case: a field handed out after others were loaded must be
        // populated too, so the document is fetched again.
        state.doc = None;
        LazyField {
            owner: self,
            name: field_info.name.clone(),
            field_num: field_info.number,
            real_value: slot,
        }
    }

    /// `getDocument()`: the stored fields of the names asked for so far,
    /// loaded on first use.
    ///
    /// # Errors
    /// A document outside the reader, or stored fields that do not decode
    /// (Java's `IllegalStateException("unable to load document")`).
    fn load(&self, state: &mut State) -> Result<()> {
        if state.doc.is_none() {
            let (segment, local) = crate::mlt::locate(self.reader, self.doc_id)?;
            let numbers: Vec<i32> = state
                .field_names
                .iter()
                .filter_map(|n| segment.field_infos().field_by_name(n).map(|i| i.number))
                .collect();
            let mut visitor = DocumentVisitor::for_fields(&numbers);
            // A segment without stored fields has no document to load.
            segment.visit_stored_document(local, &mut visitor)?;
            state.doc = Some(visitor.into_document().fields);
        }
        Ok(())
    }

    /// `fetchRealValues(name, fieldNum)`: fills every lazy value of the field
    /// with the document's value at the same position.
    fn fetch_real_values(&self, name: &str, field_num: i32) -> Result<()> {
        let mut state = self.lock();
        self.load(&mut state)?;
        let (segment, _) = crate::mlt::locate(self.reader, self.doc_id)?;
        let number = segment.field_infos().field_by_name(name).map(|i| i.number);
        let real: Vec<&StoredField> = state
            .doc
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter(|f| Some(f.field_number) == number)
            .collect();
        if let Some(lazy) = state.fields.get(&field_num) {
            for (slot, value) in lazy.iter().zip(&real) {
                *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some((*value).clone());
            }
        }
        Ok(())
    }
}

/// `LazyDocument.LazyField`: a field that reads its value from the document
/// the first time any of it is asked for.
pub struct LazyField<'a, 'r> {
    owner: &'a LazyDocument<'r>,
    name: String,
    field_num: i32,
    real_value: Slot,
}

impl LazyField<'_, '_> {
    /// `name()`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `hasBeenLoaded()`.
    pub fn has_been_loaded(&self) -> bool {
        self.real_value
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// `getRealValue()`: loads the document if this value is not yet there.
    ///
    /// # Errors
    /// What loading the document reports, and an error where Java's assertion
    /// fails: the document has fewer values of this field than were handed out.
    fn real_value(&self) -> Result<StoredField> {
        if !self.has_been_loaded() {
            self.owner.fetch_real_values(&self.name, self.field_num)?;
        }
        self.real_value
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| Error::IllegalState("field value was not lazy loaded".into()))
    }

    /// `storedValue()`.
    ///
    /// # Errors
    /// As [`Self::real_value`].
    pub fn stored_value(&self) -> Result<FieldValue> {
        Ok(self.real_value()?.value)
    }

    /// `stringValue()`: a string value, or a number's `toString()`; `None`
    /// for binary values.
    ///
    /// # Errors
    /// As [`Self::real_value`].
    pub fn string_value(&self) -> Result<Option<String>> {
        Ok(crate::mlt::stored_string_value(&self.real_value()?.value))
    }

    /// `binaryValue()`.
    ///
    /// # Errors
    /// As [`Self::real_value`].
    pub fn binary_value(&self) -> Result<Option<Vec<u8>>> {
        Ok(match self.real_value()?.value {
            FieldValue::Binary(b) => Some(b),
            _ => None,
        })
    }

    /// `numericValue()`.
    ///
    /// # Errors
    /// As [`Self::real_value`].
    pub fn numeric_value(&self) -> Result<Option<Number>> {
        Ok(match self.real_value()?.value {
            FieldValue::Int(v) => Some(Number::Int(v)),
            FieldValue::Long(v) => Some(Number::Long(v)),
            FieldValue::Float(v) => Some(Number::Float(v)),
            FieldValue::Double(v) => Some(Number::Double(v)),
            FieldValue::String(_) | FieldValue::Binary(_) => None,
        })
    }
}
