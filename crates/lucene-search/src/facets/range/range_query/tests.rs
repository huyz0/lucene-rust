//! Unit tests.

use std::sync::Arc;

use super::*;
use crate::facets::range::LongRange;
use crate::values_source::long_from_long_field;

#[test]
fn prints_as_java_and_debugs_its_check() {
    let r = LongRange::new("x", 0, true, 10, true).unwrap();
    let q = r.get_query(None, long_from_long_field("f"));
    assert_eq!(q.to_string(), "Filter(LongRange(x: 0 to 10))");
    assert_eq!(format!("{q:?}"), q.to_string());
    let check = ValueSourceQuery {
        values: RangeValues::Long {
            min: 0,
            max: 1,
            source: Arc::clone(&long_from_long_field("f")),
        },
    };
    assert_eq!(format!("{check:?}"), "ValueSourceQuery(long(f))");
}
