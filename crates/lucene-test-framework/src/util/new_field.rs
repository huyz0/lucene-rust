//! `LuceneTestCase.newField`, `newStringField` and `newTextField`: a field
//! whose type is the one asked for, randomly upgraded the first time a test
//! names the field -- stored, and for an indexed field term vectors, with
//! positions, payloads and offsets each at random -- and the same upgraded
//! type every later time (`fieldToType`, which `tearDown` clears).
//!
//! Differs: Java's map is static, shared by the whole test class and
//! cleared after each test method; here it is a [`NewFields`] the test
//! owns, and [`NewFields::clear`] is `tearDown`'s clearing.

use std::collections::HashMap;

use lucene_codecs::field_infos::IndexOptions;
use lucene_index::document::{Field, FieldType, Store, StringField, TextField};
use lucene_util::java_random::JavaRandom;

/// `fieldToType`: each field name's type, as first upgraded.
#[derive(Debug, Default)]
pub struct NewFields {
    types: HashMap<String, FieldType>,
}

impl NewFields {
    /// No field named yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `fieldToType.clear()`.
    pub fn clear(&mut self) {
        self.types.clear();
    }

    /// The type `name` was given, if it was named.
    #[must_use]
    pub fn field_type(&self, name: &str) -> Option<&FieldType> {
        self.types.get(name)
    }

    /// `newField(random, name, value, type)`.
    ///
    /// # Panics
    /// When `type` makes no field of a string value (neither indexed nor
    /// stored), as Java's `Field` constructor throws.
    pub fn new_field(
        &mut self,
        random: &mut JavaRandom,
        name: &str,
        value: &str,
        field_type: &FieldType,
    ) -> Field {
        if let Some(prev) = self.types.get(name) {
            // always use the same fieldType for the same field name
            return Field::from_string(name, value, prev.clone()).expect("a valid field");
        }
        let mut t = FieldType::copy_of(field_type);
        let set = |r: lucene_index::document::Result<()>| r.expect("an unfrozen copy");
        if !t.stored() && random.next_boolean() {
            // randomly store it
            set(t.set_stored(true));
        }
        if t.index_options() != IndexOptions::None
            && t.index_options() != IndexOptions::DocsAndCustomFreqs
            && !t.store_term_vectors()
            && random.next_boolean()
        {
            set(t.set_store_term_vectors(true));
            if !t.store_term_vector_positions() {
                let positions = random.next_boolean();
                set(t.set_store_term_vector_positions(positions));
                if positions && !t.store_term_vector_payloads() {
                    set(t.set_store_term_vector_payloads(random.next_boolean()));
                }
            }
            // a string value: offsets are allowed
            if !t.store_term_vector_offsets() {
                set(t.set_store_term_vector_offsets(random.next_boolean()));
            }
        }
        t.freeze();
        self.types.insert(name.to_string(), t.clone());
        Field::from_string(name, value, t).expect("a valid field")
    }

    /// `newStringField(random, name, value, stored)`.
    pub fn new_string_field(
        &mut self,
        random: &mut JavaRandom,
        name: &str,
        value: &str,
        stored: Store,
    ) -> Field {
        let t = match stored {
            Store::Yes => StringField::type_stored(),
            Store::No => StringField::type_not_stored(),
        };
        self.new_field(random, name, value, &t)
    }

    /// `newTextField(random, name, value, stored)`.
    pub fn new_text_field(
        &mut self,
        random: &mut JavaRandom,
        name: &str,
        value: &str,
        stored: Store,
    ) -> Field {
        let t = match stored {
            Store::Yes => TextField::type_stored(),
            Store::No => TextField::type_not_stored(),
        };
        self.new_field(random, name, value, &t)
    }
}
