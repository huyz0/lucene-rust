//! `java.util.Locale`: the constants a script passes to a case or format
//! method. None of these languages has a locale-specific case mapping, so
//! the methods that take one behave as they do without it.

use crate::object::JObject;
use crate::value::Value;
use crate::vm::{Throw, Vm};

const CLASS: &str = "java.util.Locale";

/// A locale: its language and country.
#[derive(Debug)]
struct Locale {
    language: &'static str,
    country: &'static str,
}

impl Locale {
    fn tag(&self) -> String {
        match (self.language, self.country) {
            ("", _) => "und".to_string(),
            (l, "") => l.to_string(),
            (l, c) => format!("{l}-{c}"),
        }
    }
}

impl JObject for Locale {
    fn class_name(&self) -> &str {
        CLASS
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn java_equals(&self, other: &dyn JObject) -> bool {
        other
            .as_any()
            .downcast_ref::<Locale>()
            .is_some_and(|o| o.language == self.language && o.country == self.country)
    }

    fn java_hash(&self) -> i32 {
        crate::value::string_hash(&self.java_to_string())
    }

    fn java_to_string(&self) -> String {
        match (self.language, self.country) {
            (l, "") => l.to_string(),
            (l, c) => format!("{l}_{c}"),
        }
    }

    fn call(
        &self,
        _vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        if !args.is_empty() {
            return None;
        }
        Some(Ok(Value::string(match name {
            "getLanguage" => self.language.to_string(),
            "getCountry" => self.country.to_string(),
            "toLanguageTag" => self.tag(),
            "toString" => self.java_to_string(),
            _ => return None,
        })))
    }
}

/// The locale constants.
pub fn static_field(class: &str, name: &str) -> Option<Value> {
    if class != CLASS {
        return None;
    }
    let (language, country) = match name {
        "ROOT" => ("", ""),
        "ENGLISH" => ("en", ""),
        "US" => ("en", "US"),
        "UK" => ("en", "GB"),
        "CANADA" => ("en", "CA"),
        "FRENCH" => ("fr", ""),
        "FRANCE" => ("fr", "FR"),
        "GERMAN" => ("de", ""),
        "GERMANY" => ("de", "DE"),
        _ => return None,
    };
    Some(Value::object(Locale { language, country }))
}
