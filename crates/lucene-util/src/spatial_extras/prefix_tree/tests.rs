//! Unit tests for the prefix trees' own surface: construction errors, the
//! iterator protocol, and the accessors the differential fixture
//! (`tests/spatial_prefix_tree_fixtures.rs`) does not reach.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::java_calendar::{Calendar, DAY_OF_MONTH, ERA, HOUR_OF_DAY, MONTH, YEAR};
use super::*;
use crate::spatial4j::{SpatialContextFactory, SpatialRelation};

fn geo() -> Arc<SpatialContext> {
    SpatialContext::geo_context()
}

fn flat() -> Arc<SpatialContext> {
    let mut f = SpatialContextFactory::new();
    f.geo = false;
    f.world_bounds = Some([-100.0, 100.0, -50.0, 50.0]);
    f.new_spatial_context().unwrap()
}

fn geo3d() -> Arc<SpatialContext> {
    SpatialContextFactory::geo3d()
        .new_spatial_context()
        .unwrap()
}

fn drain(mut it: Box<dyn CellIterator>) -> Vec<String> {
    let mut out = Vec::new();
    while it.has_next().unwrap() {
        out.push(it.next().unwrap().to_string());
    }
    // past the end: Java's NoSuchElementException
    assert!(matches!(it.next(), Err(Error::Runtime(m)) if m.contains("NoSuchElement")));
    out
}

/// Every tree, and the trait methods each answers the same way.
fn trees() -> Vec<Arc<dyn SpatialPrefixTree>> {
    vec![
        Arc::new(GeohashPrefixTree::new(geo(), 3).unwrap()),
        Arc::new(QuadPrefixTree::new(geo(), 4).unwrap()),
        Arc::new(QuadPrefixTree::with_bounds(flat(), [-100.0, 100.0, -50.0, 50.0], 4).unwrap()),
        Arc::new(PackedQuadPrefixTree::new(geo(), 4).unwrap()),
        Arc::new(S2PrefixTree::new(geo3d(), 3, 1).unwrap()),
        Arc::new(S2PrefixTree::new(geo3d(), 2, 2).unwrap()),
        Arc::new(DateRangePrefixTree::new(&Calendar::new_default()).unwrap()),
    ]
}

#[test]
fn every_tree_answers_the_trait() {
    for t in trees() {
        let name = t.to_string();
        assert!(!format!("{t:?}").is_empty(), "{name}");
        assert!(t.max_levels() > 0, "{name}");
        assert!(t.level_for_distance(1e-9) >= 1, "{name}");
        assert!(
            Arc::ptr_eq(t.spatial_context(), t.spatial_context()),
            "{name}"
        );
        let world = t.world_cell();
        assert_eq!(world.level(), 0, "{name}");
        let _ = format!("{world:?}"); // a legacy world cell prints as ""
        let copy: Box<dyn Cell> = world.clone();
        assert_eq!(
            copy.token_bytes_no_leaf(),
            world.token_bytes_no_leaf(),
            "{name}"
        );
        assert!(world.is_prefix_of(&*copy), "{name}");
        // 0, except S2's world cell, which Java always orders after
        assert!(world.compare_to_no_leaf(&*copy) >= 0, "{name}");
        // a child of the world, read back from its term
        let mut kids = world.next_level_cells(None).unwrap();
        assert!(kids.has_next().unwrap());
        let kid = kids.next().unwrap();
        assert_eq!(
            kids.this_cell().unwrap().token_bytes_no_leaf(),
            kid.token_bytes_no_leaf()
        );
        let back = t.read_cell(&kid.token_bytes_with_leaf()).unwrap();
        assert_eq!(
            back.token_bytes_no_leaf(),
            kid.token_bytes_no_leaf(),
            "{name}"
        );
        assert!(world.is_prefix_of(&*kid), "{name}");
        // (not `!kid.is_prefix_of(world)`: Java's packed quad compares only
        // the kid's level of bits, so its first quadrant "prefixes" the world)
        assert!(kid.compare_to_no_leaf(&*world) > 0, "{name}");
        let s2_world = world
            .as_any()
            .downcast_ref::<s2::S2PrefixTreeCell>()
            .is_some();
        assert_eq!(world.compare_to_no_leaf(&*kid) < 0, !s2_world, "{name}");
        // a tree iterator past its detail level is an error
        let shape = world.shape().unwrap();
        assert!(
            t.tree_cell_iterator(&shape, t.max_levels() + 1).is_err(),
            "{name}"
        );
        // the downcast hooks
        assert!(t.as_any().type_id() != std::any::TypeId::of::<()>());
        assert!(world.as_any().type_id() != std::any::TypeId::of::<()>());
    }
}

#[test]
fn spatial_trees_sizes_and_distances() {
    let geohash = GeohashPrefixTree::new(geo(), 3).unwrap();
    assert_eq!(geohash.world_cell().sub_cells_size(), Some(32));
    assert!(geohash.distance_for_level(2).unwrap() > 0.0);
    let p = geo().point_xy(10.0, 20.0).unwrap();
    assert_eq!(geohash.get_cell(&*p, 2).unwrap().level(), 2);

    let quad = QuadPrefixTree::new(geo(), 4).unwrap();
    assert_eq!(quad.world_cell().sub_cells_size(), Some(4));
    let qc = quad.get_cell(&*p, 9).unwrap();
    assert_eq!(qc.level(), 4, "a point's level is capped at maxLevels");
    let bad = quad.read_cell(b"AE").unwrap();
    assert!(matches!(bad.shape(), Err(Error::Runtime(m)) if m.contains("unexpected char")));
    let legacy = qc.as_any().downcast_ref::<LegacyCell>().unwrap();
    assert_eq!(legacy.bytes().len(), 4);
    assert_eq!(format!("{legacy:?}"), legacy.to_string());

    let packed = PackedQuadPrefixTree::new(geo(), 4).unwrap();
    let pc = packed.get_cell(&*p, 3);
    assert_eq!(pc.sub_cells_size(), Some(4));
    let cell = pc
        .as_any()
        .downcast_ref::<packed_quad::PackedQuadCell>()
        .unwrap();
    assert_ne!(cell.term(), 0);
    assert_eq!(cell.to_string().len(), 64);
    assert_eq!(format!("{cell:?}"), cell.to_string());
    assert_eq!(packed.world_cell().to_string(), "0".repeat(64));
    assert!(!format!("{packed:?}").is_empty());
    // a legacy cell compared with a packed one compares terms
    assert_ne!(legacy.compare_to_no_leaf(&*pc), 0);
    assert_ne!(pc.compare_to_no_leaf(&*qc), 0);
    assert!(!pc.is_prefix_of(&*qc));
    assert!(packed.distance_for_level(2).unwrap() > 0.0);
}

#[test]
fn construction_errors() {
    assert!(matches!(
        GeohashPrefixTree::new(flat(), 3),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        GeohashPrefixTree::new(geo(), 99),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        PackedQuadPrefixTree::new(geo(), packed_quad::MAX_LEVELS_POSSIBLE + 1),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        S2PrefixTree::new(geo(), 3, 1),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        S2PrefixTree::new(geo3d(), 3, 4),
        Err(Error::IllegalArgument(_))
    ));

    let ctx = geo();
    let args = |kv: &[(&str, &str)]| -> BTreeMap<String, String> {
        kv.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };
    assert!(matches!(
        make_spt(&args(&[("prefixTree", "org.example.Nope")]), &ctx),
        Err(Error::Runtime(m)) if m.contains("ClassNotFound")
    ));
    assert!(matches!(
        make_spt(&args(&[("maxLevels", "x")]), &ctx),
        Err(Error::NumberFormat(_))
    ));
    assert!(matches!(
        make_spt(&args(&[("maxDistErr", "x")]), &ctx),
        Err(Error::NumberFormat(_))
    ));
    // Java's quad tree takes any level count; the other two check theirs
    for kind in ["geohash", "packedQuad"] {
        assert!(
            make_spt(&args(&[("prefixTree", kind), ("maxLevels", "99")]), &ctx).is_err(),
            "{kind}"
        );
    }
    assert!(make_spt(&args(&[("prefixTree", "s2"), ("maxLevels", "3")]), &ctx).is_err());
    // planar without a distance: the tree's own maximum
    assert_eq!(
        make_spt(&BTreeMap::new(), &flat()).unwrap().max_levels(),
        quad::MAX_LEVELS_POSSIBLE
    );
}

#[test]
fn s2_cells() {
    let t = S2PrefixTree::new(geo3d(), 3, 2).unwrap();
    // "+" alone is the world cell, a leaf
    let w = t.read_cell(b"+").unwrap();
    assert_eq!(w.level(), 0);
    assert!(w.token_bytes_no_leaf().is_empty());
    assert_eq!(w.sub_cells_size(), Some(6));
    assert_eq!(w.to_string(), "0");
    let world = t.world_cell();
    let s2 = world
        .as_any()
        .downcast_ref::<s2::S2PrefixTreeCell>()
        .unwrap();
    assert!(s2.cell_id().is_none());
    assert_eq!(format!("{s2:?}"), s2.to_string());
    let bbox = world.shape().unwrap().bounding_box().unwrap();
    assert_eq!(bbox.min_x(), -180.0);
    let mut it = world.next_level_cells(None).unwrap();
    let face = it.next().unwrap();
    assert_eq!(face.sub_cells_size(), Some(16));
    let child = face.next_level_cells(None).unwrap().next().unwrap();
    assert!(child
        .as_any()
        .downcast_ref::<s2::S2PrefixTreeCell>()
        .unwrap()
        .cell_id()
        .is_some());
    assert!(face.is_prefix_of(&*child));
    assert!(!child.is_prefix_of(&*face));
    assert!(!child.is_prefix_of(&*world));
    assert!(world.is_prefix_of(&*child));
    assert!(child.compare_to_no_leaf(&*world) > 0);
    assert_eq!(
        world.compare_to_no_leaf(&*child),
        1,
        "Java's world cell is always after"
    );
    assert_eq!(child.token_bytes_no_leaf().len(), 2);
    assert!(matches!(t.read_cell(b"!"), Err(Error::Runtime(_))));
    assert!(matches!(
        t.read_cell(b""),
        Err(Error::ArrayIndexOutOfBounds(_))
    ));
    assert!(!format!("{t:?}").is_empty());
}

#[test]
fn iterator_protocol() {
    let t = QuadPrefixTree::new(geo(), 3).unwrap();
    let world = t.world_cell();
    let kids: Vec<Box<dyn Cell>> = {
        let mut it = world.next_level_cells(None).unwrap();
        let mut v = Vec::new();
        while it.has_next().unwrap() {
            v.push(it.next().unwrap());
        }
        v
    };
    // FilterCellIterator: without a filter, every cell; `next_from` skips
    // to the first at or after a cell; `remove` is a no-op.
    let mut f = FilterCellIterator::new(kids.clone(), None);
    assert!(f.has_next().unwrap());
    assert!(f.has_next().unwrap(), "a second hasNext keeps the cell");
    f.remove();
    let third = kids[2].clone();
    let got = f.next_from(&*third).unwrap().unwrap();
    assert_eq!(got.token_bytes_no_leaf(), third.token_bytes_no_leaf());
    assert_eq!(
        f.this_cell().unwrap().token_bytes_no_leaf(),
        third.token_bytes_no_leaf()
    );
    assert!(f.next_from(&*kids[0]).unwrap().is_some());
    assert!(f.next_from(&*kids[0]).unwrap().is_none());
    assert_eq!(drain(Box::new(f)).len(), 0);
    // with a filter, only the cells it touches, a WITHIN one made a leaf
    let shape = geo().rect(-170.0, -160.0, 10.0, 20.0).unwrap();
    let filter: Arc<dyn Shape> = shape;
    let central: Arc<dyn Shape> = geo().rect(-10.0, 10.0, -10.0, 10.0).unwrap();
    let mut g = FilterCellIterator::new(kids.clone(), Some(filter.clone()));
    let c = g.next().unwrap();
    assert_eq!(c.shape_rel(), Some(SpatialRelation::Contains));
    assert_eq!(drain(Box::new(g)).len(), 0);

    let mut s = SingletonCellIterator::new(world.clone());
    assert!(s.this_cell().is_none());
    assert!(s.has_next().unwrap());
    s.next().unwrap();
    assert_eq!(s.this_cell().unwrap().level(), 0);
    assert!(!s.has_next().unwrap());
    assert!(s.this_cell().is_none(), "hasNext clears thisCell");
    assert!(s.next().is_err());

    // TreeCellIterator: `remove` stops descent into the cell just returned
    let mut tree = TreeCellIterator::new(Some(central), 3, world.clone()).unwrap();
    assert!(!format!("{tree:?}").is_empty());
    let first = tree.next().unwrap();
    assert_eq!(first.level(), 1);
    assert_eq!(tree.this_cell().unwrap().level(), 1);
    tree.remove();
    assert!(tree.has_next().unwrap());
    assert!(tree.has_next().unwrap());
    let all = drain(Box::new(tree));
    assert!(
        !all.iter().any(|c| c.starts_with('A')),
        "never descended: {all:?}"
    );
    assert!(
        all.iter().any(|c| c.len() == 3),
        "descended elsewhere: {all:?}"
    );
}

#[test]
fn packed_quad_iterator_protocol() {
    let t = PackedQuadPrefixTree::new(geo(), 3).unwrap();
    let shape: Arc<dyn Shape> = geo().rect(-170.0, -160.0, 10.0, 20.0).unwrap();
    let mut it = t.tree_cell_iterator(&shape, 3).unwrap();
    assert!(it.has_next().unwrap());
    assert!(it.has_next().unwrap(), "a second hasNext keeps the cell");
    let c = it.next().unwrap();
    assert_eq!(
        it.this_cell().unwrap().token_bytes_no_leaf(),
        c.token_bytes_no_leaf()
    );
    assert!(!drain(it).is_empty());
    let world = t.world_cell().shape().unwrap();
    assert!(matches!(
        t.tree_cell_iterator(&world, 4),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        t.read_cell(&[1, 2]),
        Err(Error::ArrayIndexOutOfBounds(_))
    ));
}

fn date_tree() -> DateRangePrefixTree {
    DateRangePrefixTree::new(&Calendar::new_default()).unwrap()
}

fn unit(t: &DateRangePrefixTree, s: &str) -> UnitNRShape {
    match t.parse_shape(s).unwrap() {
        NRShape::Unit(u) => u,
        NRShape::Span(s) => panic!("{s:?} is a span"),
    }
}

#[test]
fn date_tree_surface() {
    let t = date_tree();
    assert_eq!(t.number_range_tree().max_levels(), 9);
    assert!(matches!(
        t.tree_level_for_calendar_field(99),
        Err(Error::IllegalArgument(_))
    ));
    assert_eq!(
        t.tree_level_for_calendar_field(YEAR as i32).unwrap(),
        YEAR_LEVEL
    );
    assert!(t.level_for_distance(1.0) == t.max_levels());
    assert!(matches!(
        t.distance_for_level(1),
        Err(Error::UnsupportedOperation(_))
    ));
    assert_eq!(t.world_cell().to_string(), "*");
    let a = unit(&t, "2014-02");
    let b = unit(&t, "2014-03");
    match t.to_range_shape(&a, &b).unwrap() {
        NRShape::Span(s) => {
            assert_eq!(s.min_unit().to_string(), "2014-02");
            assert_eq!(s.max_unit().to_string(), "2014-03");
            assert_eq!(s.levels_in_common(), YEAR_LEVEL);
            assert!(!format!("{s:?}").is_empty());
        }
        NRShape::Unit(u) => panic!("{u} is a unit"),
    }
    // a unit is a shape with no geometry
    let s = a.clone();
    assert!(s.bounding_box().is_err());
    assert!(s.area(None).is_err());
    assert!(s.center().is_err());
    assert!(s.buffered(1.0, &geo()).is_err());
    assert!(s.has_area());
    assert!(!s.is_empty());
    assert!(s.context().is_some());
    assert!(s.equals(&a.unit_clone()));
    assert!(!s.equals(&b));
    assert_eq!(s.tree().max_levels(), 9);
    assert_eq!(s.sub_cells_size(), None);
    assert!(!format!("{s:?}").is_empty());
    let mut leafy = s.clone();
    Cell::set_leaf(&mut leafy);
    assert!(Cell::is_leaf(&leafy));
    assert_eq!(
        leafy.token_bytes_with_leaf().len(),
        s.token_bytes_no_leaf().len() + 1
    );
    assert_eq!(Cell::level(&leafy), 4);
    leafy.set_shape_rel(Some(SpatialRelation::Within));
    assert_eq!(leafy.shape_rel(), Some(SpatialRelation::Within));
    assert_eq!(
        NumberRangePrefixTree::to_string_unit_raw(&t.to_unit_shape_millis(0).round_to_level(0)),
        "]"
    );
    // the span as a shape
    let span = t.parse_shape("[2014 TO 2015]").unwrap().into_shape();
    assert!(span.bounding_box().is_err());
    assert!(span.area(None).is_err());
    assert!(span.center().is_err());
    assert!(span.buffered(1.0, &geo()).is_err());
    assert!(span.has_area());
    assert!(!span.is_empty());
    assert!(span.context().is_some());
    assert!(span.equals(&*t.parse_shape("[2014 TO 2015]").unwrap().into_shape()));
    assert!(!span.equals(&a));
    // a geometric shape is no filter for a number-range cell
    let point: Arc<dyn Shape> = geo().point_xy(1.0, 2.0).unwrap();
    assert!(matches!(
        a.next_level_cells(Some(&point)),
        Err(Error::ClassCast(_))
    ));
    // the relation a cell carries for its own iteration filter
    let filter: Arc<dyn Shape> = Arc::new(unit(&t, "2014-02-05"));
    let mut it = a.next_level_cells(Some(&filter)).unwrap();
    let day = it.next().unwrap();
    let day = day.as_any().downcast_ref::<UnitNRShape>().unwrap();
    // the relation cached for the iteration's own filter, not recomputed
    assert_eq!(day.relate(&*filter).unwrap(), SpatialRelation::Within);
    assert_eq!(
        day.relate(&unit(&t, "2014-02-05")).unwrap(),
        SpatialRelation::Contains
    );
    assert!(it.has_next().is_ok());
    assert!(it.has_next().is_ok(), "a second hasNext keeps the cell");
    // a filter outside the parent: no children at all
    let elsewhere: Arc<dyn Shape> = Arc::new(unit(&t, "2015-01"));
    assert!(drain(a.next_level_cells(Some(&elsewhere)).unwrap()).is_empty());
    // a geometric shape relates to a unit by asking the shape (transposed)
    // ... which asks back: Java overflows its stack, this is an error
    assert!(
        matches!(a.relate(&*point), Err(Error::Runtime(m)) if m.contains("StackOverflowError"))
    );
    let span = t.parse_shape("[2014 TO 2015]").unwrap().into_shape();
    assert!(span.relate(&*point).is_err());
    // the guard resets: a later relation still works
    assert_eq!(a.relate(&b).unwrap(), SpatialRelation::Disjoint);
    // reading terms: empty is the world, a trailing 0 byte is a leaf
    let nr = t.number_range_tree();
    assert_eq!(nr.to_string(), "DateRangePrefixTree");
    assert_eq!(nr.read_cell(&[]).unwrap().level(), 0);
    let mut term = a.token_bytes_no_leaf();
    term.push(0);
    let leaf = nr.read_cell(&term).unwrap();
    assert!(leaf.is_leaf());
    assert_eq!(leaf.to_string(), "2014-02");
}

#[test]
fn number_range_base_errors() {
    assert!(matches!(
        NrBase::new(vec![10, 1]),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        NrBase::new(vec![1 << 15]),
        Err(Error::IllegalArgument(_))
    ));
    let base = NrBase::new(vec![300, 10]).unwrap();
    assert_eq!(base.max_term_len(), 4);
    let t = date_tree();
    // a term no level has that length
    assert!(matches!(
        t.read_cell(&[1; 40]),
        Err(Error::ArrayIndexOutOfBounds(_))
    ));
    assert!(t.number_range_tree().tree().base().max_levels() == 9);
    assert!(matches!(t.parse_shape(""), Err(Error::IllegalArgument(_))));
    let a = unit(&t, "2014-01");
    assert_eq!(a.compare_to(&unit(&t, "2014")), std::cmp::Ordering::Greater);
    assert_eq!(a.compare_to(&unit(&t, "2015")), std::cmp::Ordering::Less);
}

#[test]
fn calendar_lenient_and_cutover_edges() {
    // a month past December rolls into the next year
    let mut c = Calendar::new_default();
    c.set(YEAR, 2000);
    c.set(MONTH, 14);
    assert_eq!((c.get(YEAR), c.get(MONTH)), (2001, 2));
    // a negative hour borrows a day
    let mut c = Calendar::new_default();
    c.set(YEAR, 2000);
    c.set(HOUR_OF_DAY, -1);
    assert_eq!(
        (c.get(YEAR), c.get(MONTH), c.get(DAY_OF_MONTH)),
        (1999, 11, 31)
    );
    // a Gregorian-year date that falls before the cutover
    let mut c = Calendar::new_default();
    c.set(YEAR, 1583);
    c.set(MONTH, -10);
    assert_eq!((c.get(YEAR), c.get(MONTH)), (1582, 2));
    // a Julian-year date that falls after the cutover
    let mut c = Calendar::new_default();
    c.set(YEAR, 1581);
    c.set(MONTH, 24);
    assert_eq!((c.get(YEAR), c.get(MONTH)), (1583, 0));
    // the cutover month: October 1582 runs 1-31 but skips 5-14
    let mut c = Calendar::new_default();
    c.set(YEAR, 1582);
    c.set(MONTH, 9);
    assert_eq!(c.actual_minimum(DAY_OF_MONTH), 1);
    assert_eq!(c.actual_maximum(DAY_OF_MONTH), 31);
    assert_eq!(c.actual_maximum(MONTH), 11);
    assert_eq!(Calendar::minimum(ERA), 0);
    assert_eq!(Calendar::maximum(ERA), 1);
    assert_eq!(Calendar::maximum(YEAR), 292_278_994);
    // a cutover on January 1st, and one mid-month spanning two months
    let jan1 = Calendar::with_cutover(-12_219_292_800_000 + 78 * 86_400_000);
    let mut c = jan1.clone();
    c.set(YEAR, 1583);
    c.set(MONTH, 0);
    let _ = (
        c.actual_minimum(DAY_OF_MONTH),
        c.actual_maximum(DAY_OF_MONTH),
    );
    // a cutover year whose last Julian day is in the month before
    let mut late = Calendar::with_cutover(-12_219_292_800_000 + 20 * 86_400_000);
    late.set(YEAR, 1582);
    late.set(MONTH, 10);
    assert!(late.actual_minimum(DAY_OF_MONTH) > 1);
    let mut before = Calendar::with_cutover(-12_219_292_800_000 + 20 * 86_400_000);
    before.set(YEAR, 1582);
    before.set(MONTH, 9);
    before.set(DAY_OF_MONTH, 2);
    // October's Julian days run to the 24th, the day before the cutover
    assert_eq!(before.actual_maximum(DAY_OF_MONTH), 24);
    // the proleptic calendar's cutover year is its first: no next January
    // before it
    let mut p = Calendar::new_proleptic();
    p.set_time_in_millis(i64::MIN);
    assert_eq!(p.actual_maximum(MONTH), 11);
}

/// Index terms are read off disk: whatever their bytes -- a corrupt term,
/// or a term of a deeper or different tree -- reading the cell and asking
/// it what the traversals ask fails with an error, never a panic (which
/// could cross the FFI).
#[test]
fn arbitrary_terms_read_without_panicking() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let alphabets: [&[u8]; 4] = [
        b"ABCD+",
        b"0123456789bcdefghjkmnpqrstuvwxyz+",
        b"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*",
        &[0, 1, 2, 3, 0x7f, 0x80, 0xfe, 0xff, b'A', b'+'],
    ];
    let mut trees = trees();
    trees.push(Arc::new(PackedQuadPrefixTree::new(geo(), 2).unwrap()));
    for tree in &trees {
        let world = tree.world_cell();
        let world_shape = world.shape().unwrap();
        let mut scratch = tree.world_cell();
        for round in 0..3000 {
            let len = (next() % 40) as usize;
            let alphabet = alphabets[round % alphabets.len()];
            let term: Vec<u8> = (0..len)
                .map(|_| alphabet[(next() % alphabet.len() as u64) as usize])
                .collect();
            let _ = tree.read_cell_into(&term, &mut scratch);
            let Ok(mut cell) = tree.read_cell(&term) else {
                continue;
            };
            let _ = (cell.level(), cell.is_leaf(), cell.to_string());
            let _ = (cell.token_bytes_with_leaf(), cell.token_bytes_no_leaf());
            let _ = cell.shape().map(|s| s.bounding_box());
            let _ = cell.relate_shape(&*world_shape);
            let _ = (
                cell.compare_to_no_leaf(&*world),
                world.compare_to_no_leaf(&*cell),
            );
            let _ = (cell.is_prefix_of(&*world), world.is_prefix_of(&*cell));
            if cell.level() < tree.max_levels() {
                if let Ok(mut kids) = cell.next_level_cells(None) {
                    while let Ok(true) = kids.has_next() {
                        if kids.next().is_err() {
                            break;
                        }
                    }
                }
            }
            cell.set_leaf();
        }
    }
}

/// A quad term deeper than the tree is Java's `ArrayIndexOutOfBoundsException`
/// at the first index `makeShape` reads past `levelW` (a `C` reads none);
/// an unexpected byte first is its `RuntimeException`.
#[test]
fn too_deep_quad_terms_are_java_index_errors() {
    let quad = QuadPrefixTree::new(geo(), 2).unwrap(); // levelW has 3 entries
    let err = |term: &[u8]| {
        quad.read_cell(term)
            .unwrap()
            .shape()
            .unwrap_err()
            .to_string()
    };
    assert_eq!(err(b"ABCD"), "Index 3 out of bounds for length 3");
    assert_eq!(err(b"ABCCA"), "Index 4 out of bounds for length 3");
    assert_eq!(err(b"ABCC"), "Index 3 out of bounds for length 3");
    assert_eq!(err(b"ABxD"), "unexpected char: 120");
    let rel = quad
        .read_cell(b"AAAA")
        .unwrap()
        .relate_shape(&geo().world_bounds());
    assert!(matches!(rel, Err(Error::ArrayIndexOutOfBounds(_))));
    assert!(quad.read_cell(b"ABC").unwrap().shape().is_ok());

    let packed = PackedQuadPrefixTree::new(geo(), 2).unwrap();
    // level 5 in the low bits (`(term >> 1) & 0x1f`), past maxLevels 2
    let term = (5u64 << 1).to_be_bytes();
    let e = packed.read_cell(&term).unwrap().shape().unwrap_err();
    assert_eq!(e.to_string(), "Index 3 out of bounds for length 3");
}
/// A term past the S2 tree's 30 levels decodes, through Java's wrapping
/// shifts, to an id without a level: its token (Java's
/// `ArrayIndexOutOfBoundsException`) is empty rather than a panic.
#[test]
fn s2_term_decoding_to_no_level_has_an_empty_token() {
    let tree = S2PrefixTree::new(geo3d(), 30, 1).unwrap();
    let cell = tree.read_cell(&[b'.'; 32]).unwrap();
    assert_eq!(cell.level(), 0);
    assert!(cell.token_bytes_no_leaf().is_empty());
}
