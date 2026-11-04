use super::*;
use std::collections::HashMap;

/// `TestLabelToOrdinal`'s in-memory `LabelToOrdinalMap`.
#[derive(Default)]
struct Map {
    counter: i32,
    map: HashMap<FacetLabel, i32>,
}

impl LabelToOrdinal for Map {
    fn counter_mut(&mut self) -> &mut i32 {
        &mut self.counter
    }
    fn add_label(&mut self, label: &FacetLabel, ordinal: i32) {
        self.map.insert(label.clone(), ordinal);
    }
    fn get_ordinal(&self, label: &FacetLabel) -> i32 {
        self.map
            .get(label)
            .copied()
            .unwrap_or(Self::INVALID_ORDINAL)
    }
}

#[test]
fn ordinals_count_up() {
    let mut m = Map::default();
    for i in 0..5 {
        let label = FacetLabel::new(["d", &i.to_string()]).unwrap();
        assert_eq!(m.get_ordinal(&label), -2);
        let ord = m.next_ordinal();
        m.add_label(&label, ord);
        assert_eq!(m.get_ordinal(&label), i);
    }
    assert_eq!(m.max_ordinal(), 5);
    m.counter = i32::MAX;
    assert_eq!(m.next_ordinal(), i32::MAX);
    assert_eq!(m.max_ordinal(), i32::MAX);
}
