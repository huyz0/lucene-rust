#![cfg(test)]
use super::*;

fn square() -> Polygon {
    Polygon::new(
        &[0.0, 0.0, 1.0, 1.0, 0.0],
        &[0.0, 1.0, 1.0, 0.0, 0.0],
        vec![],
    )
    .unwrap()
}

#[test]
fn square_is_two_triangles() {
    let t = tessellate(&square(), true).unwrap();
    assert_eq!(t.len(), 2);
    for tri in &t {
        for v in 0..3 {
            assert!(tri.x(v) == 0.0 || tri.x(v) == 1.0);
            assert!(tri.y(v) == 0.0 || tri.y(v) == 1.0);
            assert_eq!(
                tri.encoded_x(v),
                GeoEncodingUtils::encode_longitude(tri.x(v)).unwrap()
            );
            assert_eq!(
                tri.encoded_y(v),
                GeoEncodingUtils::encode_latitude(tri.y(v)).unwrap()
            );
        }
    }
    assert!(t[0].to_string().contains("[true]"));
    // each triangle has two polygon edges and the shared diagonal
    let edges: usize = t
        .iter()
        .map(|tri| (0..3).filter(|&e| tri.is_edge_from_polygon(e)).count())
        .sum();
    assert_eq!(edges, 4);
}

#[test]
fn degenerate_inputs_fail_like_java() {
    let line = Polygon::new(&[0.0, 1.0, 2.0, 0.0], &[0.0, 1.0, 2.0, 0.0], vec![]).unwrap();
    assert_eq!(
        tessellate(&line, false).unwrap_err().to_string(),
        "at least three non-collinear points required"
    );
    let bow = Polygon::new(
        &[0.0, 1.0, 0.0, 1.0, 0.0],
        &[0.0, 1.0, 1.0, 0.0, 0.0],
        vec![],
    )
    .unwrap();
    assert!(tessellate(&bow, true)
        .unwrap_err()
        .to_string()
        .starts_with("Polygon self-intersection at lat="));
    let xy = XYPolygon::new(
        &[0.0, 1.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, 1.0, 1.0, 0.0],
        vec![],
    )
    .unwrap();
    assert_eq!(tessellate_xy(&xy, true).unwrap().len(), 2);
    assert!(lines_intersect(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0));
}

#[derive(Default)]
struct Recorder {
    states: Vec<String>,
    splits: Vec<String>,
}

impl Monitor for Recorder {
    fn current_state(&mut self, status: &str, points: Option<&[Point]>, tessellation: &[Triangle]) {
        self.states.push(format!(
            "{status}:{}:{}",
            points.map_or(0, <[Point]>::len),
            tessellation.len()
        ));
    }
    fn start_split(&mut self, status: &str, l: &[Point], r: &[Point]) {
        self.splits
            .push(format!("start {status} {} {}", l.len(), r.len()));
    }
    fn end_split(&mut self, status: &str) {
        self.splits.push(format!("end {status}"));
    }
}

#[test]
fn monitor_sees_every_step() {
    let mut rec = Recorder::default();
    let t = tessellate_with_monitor(&square(), false, Some(&mut rec)).unwrap();
    assert_eq!(t.len(), 2);
    assert_eq!(rec.states.first().unwrap(), "INIT:4:0");
    assert_eq!(rec.states.last().unwrap(), "COMPLETED:0:2");
    let mut rec = Recorder::default();
    let xy = XYPolygon::new(
        &[0.0, 1.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, 1.0, 1.0, 0.0],
        vec![],
    )
    .unwrap();
    tessellate_xy_with_monitor(&xy, false, Some(&mut rec)).unwrap();
    assert_eq!(rec.states.last().unwrap(), "COMPLETED:0:2");
}

#[test]
fn monitor_sees_splits() {
    // Four holes sharing one vertex: the SPLIT pass runs (and, as in
    // Java, cannot finish).
    let hole = |lats: [f64; 4], lons: [f64; 4]| Polygon::new(&lats, &lons, vec![]).unwrap();
    let p = Polygon::new(
        &[0.0, 0.0, 10.0, 10.0, 0.0],
        &[0.0, 10.0, 10.0, 0.0, 0.0],
        vec![
            hole([5.0, 5.0, 6.0, 5.0], [5.0, 7.0, 7.0, 5.0]),
            hole([5.0, 7.0, 7.0, 5.0], [5.0, 5.0, 4.0, 5.0]),
            hole([5.0, 5.0, 4.0, 5.0], [5.0, 3.0, 3.0, 5.0]),
            hole([5.0, 3.0, 3.0, 5.0], [5.0, 5.0, 6.0, 5.0]),
        ],
    )
    .unwrap();
    let mut rec = Recorder::default();
    let err = tessellate_with_monitor(&p, false, Some(&mut rec)).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Unable to Tessellate shape. Possible malformed shape detected."
    );
    assert!(rec.splits.iter().any(|s| s.starts_with("start SPLIT[1]")));
    assert!(rec.states.iter().any(|s| s.starts_with("CURE")));
    assert!(rec.states.iter().any(|s| s.starts_with("INIT[1]")));
    assert!(rec.states.iter().any(|s| s.starts_with("SPLIT[FAILED]")));
    // A corpus polygon (holes on the south pole) whose split succeeds.
    let ring = |s: &str| -> (Vec<f64>, Vec<f64>) {
        s.split(';')
            .map(|p| {
                let (a, b) = p.split_once(' ').unwrap();
                (a.parse::<f64>().unwrap(), b.parse::<f64>().unwrap())
            })
            .unzip()
    };
    let (la, lo) = ring("-86.86737826154896 26.51454402962736;-84.4712724352173 26.09376343767987;-76.25355817827216 9.181299289049523;-76.60563758407736 7.214210164740096;-79.57433616804926 3.4897899309380236;-84.40820021900568 0.6857091029569009;-85.92640268234479 0.8782562408888506;-90.0 2.8263168243244827;-90.0 4.597102435947102;-90.0 5.6391846906056;-90.0 14.105011898137173;-90.0 19.778369747749295;-90.0 22.566639414502674;-86.86737826154896 26.51454402962736");
    let (h1a, h1o) = ring("-87.49341134820244 19.979778286956353;-90.0 18.46440296320393;-90.0 21.493435169262472;-87.49341134820244 19.979778286956353");
    let (h2a, h2o) = ring("-87.21810549623036 8.141296141436479;-89.65496009215225 8.142924166776647;-90.0 7.103439437222652;-90.0 6.5683028475352625;-90.0 6.1713462579135125;-88.16473787815741 4.731716690706579;-87.21810549623036 8.141296141436479");
    let p = Polygon::new(
        &la,
        &lo,
        vec![
            Polygon::new(&h1a, &h1o, vec![]).unwrap(),
            Polygon::new(&h2a, &h2o, vec![]).unwrap(),
        ],
    )
    .unwrap();
    let mut rec = Recorder::default();
    assert!(!tessellate_with_monitor(&p, false, Some(&mut rec))
        .unwrap()
        .is_empty());
    assert!(rec.splits.iter().any(|s| s == "end SPLIT[1]"));
    // A cartesian polygon off the globe cannot be reported as Points.
    let xy = XYPolygon::new(
        &[0.0, 1.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, 100.0, 100.0, 0.0],
        vec![],
    )
    .unwrap();
    let mut rec = Recorder::default();
    assert!(tessellate_xy_with_monitor(&xy, false, Some(&mut rec))
        .unwrap_err()
        .to_string()
        .starts_with("invalid latitude 100.0"));
}
