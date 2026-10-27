//! Actual-Java paired BaseXY generator, hook and oracle workloads.
use lucene_test_framework::geo::base_xy_point::{
    cartesian_distance, rect_contains_point, BaseXYPoint, XYPointCase,
};
use lucene_util::geo::{GeoError, XYGeometry, XYPolygon, XYRectangle};
use lucene_util::java_random::JavaRandom;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn step(d: u64, value: u64) -> u64 {
    d.wrapping_mul(31).wrapping_add(value)
}
fn values(mut d: u64, values: &[f32]) -> u64 {
    d = step(d, values.len() as u64);
    for value in values {
        d = step(d, u64::from(value.to_bits()));
    }
    d
}
fn polygon(d: u64, value: &XYPolygon) -> u64 {
    assert_eq!(value.num_holes(), 0);
    step(values(values(d, value.poly_x()), value.poly_y()), 0)
}
fn rectangle(d: u64, value: &XYRectangle) -> u64 {
    values(d, &[value.min_x, value.max_x, value.min_y, value.max_y])
}
fn geometries(mut d: u64, values: &[XYGeometry]) -> u64 {
    d = step(d, values.len() as u64);
    for value in values {
        d = match value {
            XYGeometry::Point(point) => self::values(step(d, 0), &[point.x(), point.y()]),
            XYGeometry::Rectangle(value) => rectangle(step(d, 1), value),
            XYGeometry::Polygon(value) => polygon(step(d, 2), value),
            other => panic!("unexpected default geometry {other:?}"),
        };
    }
    d
}
#[derive(Default)]
struct Capture {
    digest: u64,
}
impl Capture {
    fn capture(&mut self, callback: u64, xs: &[f32], ys: &[f32]) -> Result<(), GeoError> {
        assert_eq!(xs.len(), ys.len());
        self.digest = values(values(step(self.digest, callback), xs), ys);
        Ok(())
    }
}
impl XYPointCase for Capture {
    type Error = GeoError;
    fn verify_random_rectangles(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.capture(1, xs, ys)
    }
    fn verify_random_distances(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.capture(2, xs, ys)
    }
    fn verify_random_polygons(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.capture(3, xs, ys)
    }
    fn verify_random_geometries(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.capture(4, xs, ys)
    }
}
fn batch(
    which: usize,
    random: &mut JavaRandom,
    capture: &mut Capture,
    oracle: &XYRectangle,
) -> u64 {
    random.set_seed(42);
    capture.digest = 0;
    let mut d = 0;
    match which {
        0..=5 => {
            let mut base = BaseXYPoint::new(capture, random, 1);
            match which {
                0 => base.random_tiny(),
                1 => base.random_medium(),
                2 => base.same_point_many_times(),
                3 => base.low_cardinality(),
                4 => base.all_x_equal(),
                _ => base.all_y_equal(),
            }
            .unwrap();
            d = capture.digest;
        }
        6 => {
            for _ in 0..16 {
                d = step(d, u64::from(capture.next_x(random).to_bits()));
                d = step(d, u64::from(capture.next_y(random).to_bits()));
                d = rectangle(d, &capture.next_box(random));
                d = polygon(d, &capture.next_polygon(random, false).unwrap());
                d = geometries(d, &capture.next_geometry(random, false).unwrap());
            }
        }
        7 => {
            for at in 0..1024 {
                let x = f64::from(at % 17) / 4.0 - 2.0;
                let y = f64::from(at % 19) / 4.0 - 1.0;
                d = step(d, u64::from(rect_contains_point(oracle, x, y)));
                d = step(
                    d,
                    cartesian_distance(
                        x,
                        y,
                        f64::from(at % 13) / 8.0 - 1.0,
                        f64::from(at % 11) / 8.0 - 1.0,
                    )
                    .to_bits(),
                );
            }
        }
        _ => unreachable!(),
    }
    step(d, u64::from(random.next_int() as u32))
}
fn run(mut f: impl FnMut() -> u64, millis: u64, expected: u64) -> (f64, u64) {
    let start = Instant::now();
    let mut count = 0;
    loop {
        let actual = f();
        assert_eq!(actual, expected, "complete batch digest changed");
        black_box(actual);
        count += 1;
        if start.elapsed() >= Duration::from_millis(millis) {
            break;
        }
    }
    (start.elapsed().as_nanos() as f64 / count as f64, count)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args[1].parse().unwrap();
    let measure = args[2].parse().unwrap();
    let mut random = JavaRandom::new(42);
    let mut capture = Capture::default();
    let oracle = XYRectangle::new(-2.0, 3.0, -1.0, 4.0).unwrap();
    let example = XYPolygon::new(&[0., 3., 3., 1., 0.], &[0., 0., 2., 4., 0.], vec![]).unwrap();
    println!(
        "base_xy_capture_owned_bytes\t{}",
        std::mem::size_of::<Capture>()
    );
    println!(
        "base_xy_1000_point_arrays_owned_bytes\t{}",
        2 * std::mem::size_of::<Vec<f32>>() + 2000 * std::mem::size_of::<f32>()
    );
    println!(
        "base_xy_polygon_owned_bytes\t{}",
        std::mem::size_of::<XYPolygon>() + 2 * example.num_points() * std::mem::size_of::<f32>()
    );
    for (which, name) in [
        "base_xy_random_tiny",
        "base_xy_random_medium",
        "base_xy_same_point",
        "base_xy_low_cardinality",
        "base_xy_all_x",
        "base_xy_all_y",
        "base_xy_shape_hooks",
        "base_xy_oracles",
    ]
    .iter()
    .enumerate()
    {
        let expected = batch(which, &mut random, &mut capture, &oracle);
        println!("#check\t{name}\t{expected}\t1");
        run(
            || {
                batch(
                    black_box(which),
                    &mut random,
                    &mut capture,
                    black_box(&oracle),
                )
            },
            warmup,
            expected,
        );
        let (ns, count) = run(
            || {
                batch(
                    black_box(which),
                    &mut random,
                    &mut capture,
                    black_box(&oracle),
                )
            },
            measure,
            expected,
        );
        println!("{name}\t{ns:.3}\t{count}");
    }
}
