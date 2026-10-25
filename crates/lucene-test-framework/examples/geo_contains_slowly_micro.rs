//! Actual GeoTestUtil helper pair; setup and analytic expected results are untimed.
use lucene_test_framework::geo::geo_test_util::contains_slowly;
use lucene_util::geo::Polygon;
use std::{
    hint::black_box,
    mem::size_of,
    time::{Duration, Instant},
};

const QUERY_COUNT: usize = 128;
// The measured implementation borrows both slices. This source-level payload
// estimate is not allocator instrumentation or a peak-memory measurement.
const COORDINATE_COPIES_PER_CALL: usize = 0;
const CORNERS: [(i32, i32); 7] = [
    (-176, -80),
    (176, -80),
    (176, -16),
    (-48, -16),
    (-48, 80),
    (-176, 80),
    (-176, -80),
];

struct Input<'a> {
    name: &'static str,
    polygon: &'a Polygon,
    queries: Vec<(f64, f64)>,
    expected: u64,
}

fn polygon(subdivided: bool) -> Polygon {
    let count = if subdivided { 1025 } else { CORNERS.len() };
    let mut latitudes = Vec::with_capacity(count);
    let mut longitudes = Vec::with_capacity(count);
    for edge in CORNERS.windows(2) {
        let (x, y) = edge[0];
        let (to_x, to_y) = edge[1];
        let dx = (to_x - x).signum();
        let dy = (to_y - y).signum();
        let length = (to_x - x).abs() + (to_y - y).abs();
        let points = if subdivided { length } else { 1 };
        for step in 0..points {
            longitudes.push(f64::from(x + step * dx));
            latitudes.push(f64::from(y + step * dy));
        }
    }
    longitudes.push(f64::from(CORNERS[0].0));
    latitudes.push(f64::from(CORNERS[0].1));
    assert_eq!(latitudes.len(), count);
    Polygon::new(&latitudes, &longitudes, vec![]).unwrap()
}

// Independent analytic membership for this axis-aligned L, including its boundary.
fn expected(latitude: f64, longitude: f64) -> bool {
    (-80.0..=80.0).contains(&latitude)
        && (-176.0..=176.0).contains(&longitude)
        && (latitude <= -16.0 || longitude <= -48.0)
}

fn input<'a>(name: &'static str, polygon: &'a Polygon, kind: usize) -> Input<'a> {
    let mut queries = Vec::with_capacity(QUERY_COUNT);
    let mut digest = 0u64;
    for at in 0..QUERY_COUNT {
        let jitter = (at % 8) as f64 * 0.25;
        let (latitude, longitude) = match kind {
            0 => match at % 4 {
                0 => (-40.0 + jitter, -20.0 + jitter),
                1 => (20.0 + jitter, -100.0 + jitter),
                2 => (20.0 + jitter, 20.0 + jitter),
                _ => (60.0 + jitter, 100.0 + jitter),
            },
            1 => match at % 4 {
                0 => (-81.0, 0.0),
                1 => (81.0, 0.0),
                2 => (0.0, -177.0),
                _ => (0.0, 177.0),
            },
            _ => (-80.0, -176.0 + (at % 3 + 1) as f64 * 0.25),
        };
        queries.push((latitude, longitude));
        digest = digest
            .wrapping_mul(31)
            .wrapping_add(u64::from(expected(latitude, longitude)));
    }
    Input {
        name,
        polygon,
        queries,
        expected: digest,
    }
}

fn batch(input: &Input<'_>) -> u64 {
    let mut digest = 0u64;
    for &(latitude, longitude) in black_box(input.queries.as_slice()) {
        // Keep the polygon and queries opaque so repeated immutable calls cannot
        // be hoisted out of the timed batch after the borrowed-slice optimisation.
        let inside = contains_slowly(black_box(input.polygon), latitude, longitude).unwrap();
        digest = digest.wrapping_mul(31).wrapping_add(u64::from(inside));
    }
    digest
}

fn run(input: &Input<'_>, millis: u64) -> (f64, u64) {
    let started = Instant::now();
    let mut count = 0;
    loop {
        let actual = batch(input);
        assert_eq!(
            actual, input.expected,
            "{} analytic digest differs",
            input.name
        );
        black_box(actual);
        count += 1;
        if started.elapsed() >= Duration::from_millis(millis) {
            break;
        }
    }
    (started.elapsed().as_nanos() as f64 / count as f64, count)
}

fn memory(size: &str, polygon: &Polygon) {
    let payload = 2 * polygon.num_points() * size_of::<f64>();
    println!("geo_contains_{size}_polygon_coordinate_payload_bytes\t{payload}");
    println!(
        "geo_contains_{size}_polygon_retained_estimated_bytes\t{}",
        size_of::<Polygon>() + payload
    );
    println!(
        "geo_contains_{size}_per_call_temporary_coordinate_payload_bytes\t{}",
        COORDINATE_COPIES_PER_CALL * polygon.num_points() * size_of::<f64>()
    );
    println!(
        "geo_contains_{size}_per_call_temporary_arrays_estimated_bytes\t{}",
        COORDINATE_COPIES_PER_CALL * polygon.num_points() * size_of::<f64>()
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args[1].parse().unwrap();
    let measure = args[2].parse().unwrap();
    assert!(measure > 0);
    let small = polygon(false);
    let large = polygon(true);
    println!("#memory_scope\tretained polygon inline size plus coordinate payload; empty holes own no allocation; query arrays excluded");
    println!("#memory_scope\ttemporary coordinate payload is the declared implementation's source-level clone estimate; allocator metadata/peak RSS excluded");
    memory("small", &small);
    memory("large", &large);
    let inputs = [
        input("geo_contains_small_full_scan", &small, 0),
        input("geo_contains_large_full_scan", &large, 0),
        input("geo_contains_large_bbox_rejection", &large, 1),
        input("geo_contains_large_boundary_early_exit", &large, 2),
    ];
    for input in inputs {
        println!(
            "#check\t{}\t{}\t{QUERY_COUNT}\t{}",
            input.name,
            input.expected,
            input.polygon.num_points()
        );
        run(&input, warmup);
        let (ns, count) = run(&input, measure);
        println!("{}\t{ns:.3}\t{count}", input.name);
    }
}
