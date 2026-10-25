//! Paired actual-Java ShapeTestUtil workloads, complete raw-coordinate digests.
use lucene_test_framework::geo::shape_test_util::*;
use lucene_util::{geo::XYPolygon, java_random::JavaRandom};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
fn step(d: u64, v: u64) -> u64 {
    d.wrapping_mul(31).wrapping_add(v)
}
fn values(mut d: u64, v: &[f32]) -> u64 {
    d = step(d, v.len() as u64);
    for value in v {
        d = step(d, u64::from(value.to_bits()));
    }
    d
}
fn polygon(d: u64, p: &XYPolygon) -> u64 {
    values(values(d, p.poly_x()), p.poly_y())
}
fn batch(case: usize, oracle: &XYPolygon) -> u64 {
    let mut random = JavaRandom::new(42);
    let mut d = 0;
    match case {
        0 => {
            for _ in 0..512 {
                d = step(d, u64::from(next_float(&mut random).to_bits()));
            }
        }
        1 => {
            for _ in 0..32 {
                let b = next_box(&mut random);
                d = values(d, &[b.min_x, b.max_x, b.min_y, b.max_y]);
                let p = next_xy_point(&mut random);
                d = values(d, &[p.x(), p.y()]);
                let c = next_circle(&mut random);
                d = values(d, &[c.x(), c.y(), c.radius()]);
            }
        }
        2 => {
            for _ in 0..8 {
                d = polygon(d, &next_polygon(&mut random, false).unwrap());
                let l = next_line(&mut random, false).unwrap();
                d = values(values(d, l.x()), l.y());
            }
        }
        3 => {
            for _ in 0..16 {
                d = polygon(d, &create_regular_polygon(1.0, 7.0, 5.0, 64).unwrap());
            }
        }
        4 => {
            for at in 0..1024 {
                d = step(
                    d,
                    u64::from(
                        contains_slowly(
                            oracle,
                            f64::from(at % 17) / 4.0 - 1.0,
                            f64::from(at % 19) / 4.0 - 1.0,
                        )
                        .unwrap(),
                    ),
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
        assert_eq!(actual, expected);
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
    let oracle = XYPolygon::new(&[0., 3., 3., 1., 0.], &[0., 0., 2., 4., 0.], vec![]).unwrap();
    println!(
        "shape_test_util_rng_owned_bytes\t{}",
        std::mem::size_of::<JavaRandom>()
    );
    println!(
        "shape_test_util_polygon_owned_bytes\t{}",
        std::mem::size_of::<XYPolygon>() + 2 * oracle.num_points() * std::mem::size_of::<f32>()
    );
    for (case, name) in [
        "shape_biased_floats",
        "shape_primitives",
        "shape_polygon_line",
        "shape_regular_polygon",
        "shape_contains_slowly",
    ]
    .iter()
    .enumerate()
    {
        let expected = batch(case, &oracle);
        println!("#check\t{name}\t{expected}\t1");
        run(|| batch(case, &oracle), warmup, expected);
        let (ns, count) = run(|| batch(case, &oracle), measure, expected);
        println!("{name}\t{ns:.3}\t{count}");
    }
}
