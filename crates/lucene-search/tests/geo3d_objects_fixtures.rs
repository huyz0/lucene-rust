//! geo3d shapes' and `PointInGeo3DShapeQuery`'s `equals`, `hashCode` and
//! `toString` (and the query's `ramBytesUsed` and `visit`), differentially
//! against Lucene 10.5.0: `fixtures/src/GenGeo3dObjects.java` read back
//! every shape of `GenGeo3d`'s corpus from its serialized bytes and recorded
//! what Java answers. This reads the same bytes and requires the same
//! strings, the same hashes and, over every pair of shapes, the same
//! equality.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use lucene_search::document::geo::geo3d::PointInGeo3DShapeQuery;
use lucene_search::extended_query::DocumentClause;
use lucene_search::query::Clause;
use lucene_search::query_visitor::{visit, QueryVisitor};
use lucene_util::spatial3d::serializable::Input;
use lucene_util::spatial3d::standard_objects::{read_planet_object, StandardObject};
use lucene_util::spatial3d::GeoShape;

fn read(file: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/geo3d")
            .join(file),
    )
    .expect("run scripts/gen-fixtures.sh --only GenGeo3dObjects")
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn unesc(s: &str) -> String {
    s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\\", "\\")
}

/// A shape read back from its bytes; `None` for a solid (not a shape).
fn shape(bytes: &[u8]) -> Option<Arc<dyn GeoShape>> {
    let object = read_planet_object(&mut Input::new(bytes)).expect("bytes Java wrote");
    Some(match object {
        StandardObject::Polygon(s) => s,
        StandardObject::PointShape(s) => s,
        StandardObject::BBox(s) => s,
        StandardObject::Circle(s) => s,
        StandardObject::Path(s) => s,
        StandardObject::AreaShape(s) => s,
        StandardObject::MembershipShape(s) => s,
        StandardObject::Solid(_) => return None,
        StandardObject::Point(_) => panic!("a point is not a planet object"),
    })
}

/// Every shape of the corpus by id, in the corpus's order.
fn corpus() -> Vec<(u32, Arc<dyn GeoShape>)> {
    let mut shapes = Vec::new();
    for line in read("shapes.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] != "SH" {
            continue;
        }
        let arrow = f.iter().position(|&x| x == "=>").unwrap();
        if f[arrow + 1] != "OK" {
            continue;
        }
        if let Some(s) = shape(&unhex(f[arrow + 3])) {
            shapes.push((f[1].parse().unwrap(), s));
        }
    }
    shapes
}

#[test]
fn every_shape_is_equal_hashed_and_printed_as_java() {
    let shapes = corpus();
    let index: HashMap<u32, usize> = shapes
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, i))
        .collect();
    let mut seen = 0;
    let mut equal_pairs = 0;
    for line in read("objects.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] != "OBJ" {
            continue;
        }
        let id: u32 = f[1].parse().unwrap();
        let s = &shapes[index[&id]].1;
        assert_eq!(s.java_hash_code(), f[3].parse::<i32>().unwrap(), "{id}");
        assert_eq!(s.java_to_string(), unesc(f[5]), "{id}");
        assert_eq!(format!("{s}"), unesc(f[5]));
        let want: BTreeSet<u32> = f[4].split(',').map(|x| x.parse().unwrap()).collect();
        assert!(want.contains(&id), "a shape equals itself");
        for (other_id, other) in &shapes {
            assert_eq!(
                s.java_equals(&**other),
                want.contains(other_id),
                "{id} equals {other_id}"
            );
        }
        equal_pairs += want.len();
        seen += 1;
    }
    assert_eq!(seen, shapes.len());
    assert_eq!(seen, 1508);
    // Some distinct records build equal shapes.
    assert!(equal_pairs > seen);
}

/// Records the leaves a visit reaches.
struct Leaves {
    field: Option<&'static str>,
    leaves: usize,
}

impl QueryVisitor for Leaves {
    fn visit_leaf(&mut self, _query: &Clause) {
        self.leaves += 1;
    }

    fn accept_field(&self, field: &str) -> bool {
        self.field.is_none_or(|f| f == field)
    }
}

#[test]
fn every_query_is_equal_hashed_printed_and_sized_as_java() {
    let shapes: HashMap<u32, Arc<dyn GeoShape>> = corpus().into_iter().collect();
    let long = "a_rather_longer_geo3d_field_name";
    let mut strings = 0;
    let mut seen = 0;
    for line in read("objects.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] != "Q" {
            continue;
        }
        let id: u32 = f[1].parse().unwrap();
        let shape = &shapes[&id];
        let q = PointInGeo3DShapeQuery::new("f", shape.clone()).expect("Java built it");
        assert_eq!(q.java_hash_code(), f[2].parse::<i32>().unwrap(), "{id}");
        assert_eq!(q.ram_bytes_used(), f[3].parse::<u64>().unwrap());
        let long_q = PointInGeo3DShapeQuery::new(long, shape.clone()).unwrap();
        assert_eq!(long_q.ram_bytes_used(), f[4].parse::<u64>().unwrap());
        if f.len() > 5 {
            assert_eq!(q.to_string_with_field("f"), unesc(f[5]));
            assert_eq!(q.to_string_with_field("g"), unesc(f[6]));
            strings += 1;
        }
        assert_eq!(
            q.to_string_with_field("f"),
            format!("PointInGeo3DShapeQuery: Shape: {shape}")
        );
        assert!(q == PointInGeo3DShapeQuery::new("f", shape.clone()).unwrap());
        assert!(q != long_q, "another field");
        assert_eq!(q.field(), "f");
        assert!(Arc::ptr_eq(q.shape(), shape));

        let clause: Clause = DocumentClause::new(format!("q{id}"), Arc::new(q)).into();
        for (field, leaves) in [(None, 1), (Some("f"), 1), (Some("g"), 0)] {
            let mut v = Leaves { field, leaves: 0 };
            visit(&clause, &mut v);
            assert_eq!(v.leaves, leaves, "{field:?}");
        }
        seen += 1;
    }
    assert_eq!(strings, 40);
    assert_eq!(seen, 1508);
}
