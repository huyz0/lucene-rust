#![cfg(test)]
use super::*;

/// The tree walk as Java (and this port, before the bounding-box
/// shortcut) writes it: every node tests its whole subtree's `isWithin`.
fn java_distance(
    c: &Component,
    parts: &Parts,
    style: DistanceStyle,
    x: f64,
    y: f64,
    z: f64,
) -> f64 {
    match c {
        Component::Node(n) => {
            if !c.is_within(parts, x, y, z) {
                return f64::INFINITY;
            }
            min(
                java_distance(&n.child1, parts, style, x, y, z),
                java_distance(&n.child2, parts, style, x, y, z),
            )
        }
        Component::Endpoint(i) => parts.endpoints[*i].distance(parts, style, x, y, z),
        Component::Segment(i) => parts.segments[*i].distance(parts, style, x, y, z),
    }
}

fn java_delta(c: &Component, parts: &Parts, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
    match c {
        Component::Node(n) => {
            if !c.is_within(parts, x, y, z) {
                return f64::INFINITY;
            }
            min(
                java_delta(&n.child1, parts, style, x, y, z),
                java_delta(&n.child2, parts, style, x, y, z),
            )
        }
        Component::Endpoint(i) => parts.endpoints[*i].path_delta_distance(style, x, y, z),
        Component::Segment(i) => parts.segments[*i].path_delta_distance(parts, style, x, y, z),
    }
}

/// The minimum over every segment and endpoint, no tree at all.
fn brute_distance(parts: &Parts, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
    let segments = parts
        .segments
        .iter()
        .map(|s| s.distance(parts, style, x, y, z));
    let endpoints = parts
        .endpoints
        .iter()
        .map(|e| e.distance(parts, style, x, y, z));
    segments.chain(endpoints).fold(f64::INFINITY, min)
}

struct Rng(u64);

impl Rng {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn bbox_tree_walk_matches_java_walk_and_brute_force_bit_for_bit() {
    let styles = [
        DistanceStyle::Arc,
        DistanceStyle::Linear,
        DistanceStyle::LinearSquared,
        DistanceStyle::Normal,
        DistanceStyle::NormalSquared,
    ];
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (mut inside, mut total) = (0usize, 0usize);
    for pm in [PlanetModel::wgs84(), PlanetModel::sphere()] {
        for n in [1usize, 2, 3, 7, 33] {
            let width = 0.001 + 0.05 * rng.f();
            let pts: Vec<GeoPoint> = (0..n)
                .map(|i| {
                    let t = i as f64 / n as f64;
                    let lat = -0.5 + t + 0.1 * (t * 13.0).sin();
                    let lon = -1.0 + 1.5 * t + 0.05 * rng.f();
                    GeoPoint::from_lat_lon(&pm, lat, lon).unwrap()
                })
                .collect();
            let path = GeoStandardPath::new(&pm, width, &pts).unwrap();
            let root = path.root_component.as_ref().unwrap();
            let parts = &path.parts;
            for i in 0..2000 {
                // Half near the path's points, half anywhere near it.
                let base = &pts[i % n];
                let spread = if i % 2 == 0 { width * 2.0 } else { 0.6 };
                let p = GeoPoint::from_lat_lon(
                    &pm,
                    base.latitude() + spread * (rng.f() - 0.5),
                    base.longitude() + spread * (rng.f() - 0.5),
                )
                .unwrap();
                let (x, y, z) = (p.x, p.y, p.z);
                total += 1;
                inside += usize::from(path.is_within_xyz(x, y, z));
                for style in styles {
                    let got = root.distance(parts, style, x, y, z);
                    let java = java_distance(root, parts, style, x, y, z);
                    let brute = brute_distance(parts, style, x, y, z);
                    assert_eq!(got.to_bits(), java.to_bits(), "{style:?} {p}");
                    assert_eq!(got.to_bits(), brute.to_bits(), "{style:?} {p}");
                    let got = root.path_delta_distance(parts, style, x, y, z);
                    let java = java_delta(root, parts, style, x, y, z);
                    assert_eq!(got.to_bits(), java.to_bits(), "{style:?} {p}");
                }
            }
        }
    }
    // The corpus exercises both sides of the path's boundary.
    assert!(
        inside > total / 10 && inside < total * 9 / 10,
        "{inside}/{total}"
    );
}

/// The bounding-box-pruned `intersects` against Java's full walk, on
/// random paths (equator and meridian ones too, whose planes are axis
/// parallel, as every solid face is) and random planes: solid faces
/// bounded by the other faces, and arbitrary planes with random bounds.
#[test]
fn pruned_intersects_matches_the_full_walk() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut hits, mut calls, mut pruned_leaves) = (0usize, 0usize, 0usize);
    for pm in [PlanetModel::wgs84(), PlanetModel::sphere()] {
        for shape in 0..12 {
            let n = [1usize, 2, 3, 5, 17, 40][shape % 6];
            let width = match shape % 4 {
                0 => 1e-6,
                1 => 0.01 + 0.05 * rng.f(),
                2 => 0.3,
                _ => 1.5,
            };
            let pts: Vec<GeoPoint> = (0..n)
                .map(|i| {
                    let t = i as f64 / n as f64;
                    let (lat, lon) = match shape % 3 {
                        // Along the equator / a meridian: axis-parallel
                        // connecting planes.
                        0 => (0.0, -1.0 + 2.0 * t),
                        1 => (-1.2 + 2.4 * t, 0.5),
                        _ => (
                            -0.5 + t + 0.1 * (t * 13.0).sin(),
                            -1.0 + 1.5 * t + 0.05 * rng.f(),
                        ),
                    };
                    GeoPoint::from_lat_lon(&pm, lat, lon).unwrap()
                })
                .collect();
            let Ok(path) = GeoStandardPath::new(&pm, width, &pts) else {
                continue;
            };
            let root = path.root_component.as_ref().unwrap();
            let parts = &path.parts;
            let boxes = parts.leaf_boxes();
            let mut check = |p: &Plane, notable: &[GeoPoint], bounds: &[&dyn Membership]| {
                let full = root.intersects(parts, p, notable, bounds);
                let probe = PlaneProbe::new(p);
                let pruned = root.intersects_pruned(parts, boxes, &probe, notable, bounds, false);
                assert_eq!(full, pruned, "{p:?} on {}", path.java_to_string());
                calls += 1;
                hits += usize::from(full);
                pruned_leaves += (0..parts.segments.len())
                    .filter(|&i| probe.misses(&boxes[0][i]) && parts.segments[i].steep_to(&probe))
                    .count();
            };
            for i in 0..400 {
                let base = &pts[i % n];
                let spread = if i % 2 == 0 { width * 2.0 } else { 0.8 };
                let c = GeoPoint::from_lat_lon(
                    &pm,
                    (base.latitude() + spread * (rng.f() - 0.5)).clamp(-1.5, 1.5),
                    base.longitude() + spread * (rng.f() - 0.5),
                )
                .unwrap();
                // A solid around `c`, extent log-uniform in [1e-7, 1].
                let s = 10f64.powf(-7.0 + 7.0 * rng.f());
                let lo = [c.x - s * rng.f(), c.y - s * rng.f(), c.z - s * rng.f()];
                let hi = [lo[0] + s, lo[1] + s, lo[2] + s];
                let face = |axis: usize, at: f64, inside: f64| {
                    let mut v = [0.0; 3];
                    v[axis] = 1.0;
                    let mut ip = [0.0; 3];
                    ip[axis] = inside;
                    SidedPlane::from_xyz_normal(
                        ip[0],
                        ip[1],
                        ip[2],
                        &Vector::new(v[0], v[1], v[2]),
                        -at,
                    )
                    .unwrap()
                };
                let faces: Vec<SidedPlane> = (0..3)
                    .flat_map(|a| [face(a, lo[a], hi[a]), face(a, hi[a], lo[a])])
                    .collect();
                for f in 0..6 {
                    let others: Vec<&dyn Membership> = (0..6)
                        .filter(|&g| g != f)
                        .map(|g| &faces[g] as &dyn Membership)
                        .collect();
                    check(&faces[f], &[], &others);
                    check(&faces[f], &path.edge_points, &[]);
                }
                // An arbitrary plane near `c`, bounded by two random
                // sided planes through the origin.
                let v = Vector::new(rng.f() - 0.5, rng.f() - 0.5, rng.f() - 0.5);
                if let Some(nv) = v.normalize() {
                    let d = -(nv.x * c.x + nv.y * c.y + nv.z * c.z) + 0.01 * (rng.f() - 0.5);
                    let p = Plane::with_d(&nv, d);
                    let b1 = SidedPlane::from_vectors(&c, &pts[0], base);
                    let b2 = SidedPlane::from_vectors(&pts[0], &c, &pts[n - 1]);
                    check(&p, &[], &[]);
                    if let (Ok(b1), Ok(b2)) = (b1, b2) {
                        check(&p, &[], &[&b1, &b2]);
                    }
                }
            }
        }
    }
    // Both answers occur, and the prune does fire.
    assert!(hits > calls / 50 && hits < calls / 2, "{hits}/{calls}");
    assert!(pruned_leaves > calls, "{pruned_leaves} over {calls} calls");
}

#[test]
fn full_path_distance_cache_returns_the_computed_value() {
    let pm = PlanetModel::sphere();
    let a = GeoPoint::from_lat_lon(&pm, 0.0, 0.0).unwrap();
    let b = GeoPoint::from_lat_lon(&pm, 0.0, 0.25).unwrap();
    let path = GeoStandardPath::new(&pm, 0.01, &[a.clone(), b.clone()]).unwrap();
    let seg = &path.parts.segments[0];
    for style in [DistanceStyle::Arc, DistanceStyle::LinearSquared] {
        let want = style.to_aggregation_form(style.compute_distance(&a, b.x, b.y, b.z));
        assert_eq!(seg.full_path_distance(style).to_bits(), want.to_bits());
        // Cached: the same bits again.
        assert_eq!(seg.full_path_distance(style).to_bits(), want.to_bits());
    }
}
