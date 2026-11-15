//! The text a connector template is made of: the JSON reader and writers, JSONPath, and the
//! `${...}` substitution that every request body and header goes through. Any text must parse or be
//! refused, never panic, recurse without bound or expand without limit.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lucene_ml::remote::{json::Json, jsonpath::JsonPath, substitute};

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let (path, doc) = text.split_once('\n').unwrap_or(("$", &text));
    for parsed in [Json::parse(doc), Json::parse_lenient(doc)].into_iter().flatten() {
        let _ = parsed.to_compact();
        let _ = parsed.to_gson_doubles();
        if let Ok(p) = JsonPath::compile(path) {
            let _ = p.is_definite();
            let _ = p.read(&parsed);
        }
    }
    let values: Vec<(String, Option<String>)> =
        vec![("parameters.a".into(), Some("${parameters.b}".into())), ("parameters.b".into(), Some("${parameters.a}".into())), ("parameters.c".into(), None)];
    let _ = substitute::substitute(&text, "parameters", &values);
    let _ = substitute::substitute(&text, "credential", &values);
});
