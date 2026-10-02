//! Lucene's spatial prefix trees, the date-range tree and the spatial query
//! arguments, differentially against Lucene 10.5.0 (with Spatial4j 0.8 and
//! s2-geometry 1.0.0): `fixtures/src/GenSpatialPrefixTree.java` built twelve
//! trees through `SpatialPrefixTreeFactory.makeSPT` (geohash, quad and
//! packed quad on geodetic, planar and Geo3D contexts, S2 at arity 1 and 2,
//! and the factory's defaults) and recorded their levels and distances, the
//! cells each yields for random shapes at random detail (token, level, leaf
//! flag and relation, in order), cells read back from terms with their
//! shapes, prefix relations, children and relations; then the date-range
//! tree on both calendars (parsing, formatting, round trips, sub-cell
//! counts, range shapes and their relations and cells, terms read back),
//! and `SpatialArgsParser`/`SpatialOperation` on random shapes. This test
//! repeats every call and compares the formatted results.

use std::collections::BTreeMap;
use std::sync::Arc;

use lucene_util::spatial4j::{Error, Shape, SpatialContext, SpatialContextFactory};
use lucene_util::spatial_extras::prefix_tree::java_calendar::{
    Calendar, DAY_OF_MONTH, ERA, HOUR_OF_DAY, MILLISECOND, MINUTE, MONTH, SECOND, YEAR,
};
use lucene_util::spatial_extras::prefix_tree::{
    make_spt, Cell, CellIterator, DateRangePrefixTree, NRShape, PackedQuadPrefixTree, S2PrefixTree,
    SpatialPrefixTree, UnitNRShape,
};
use lucene_util::spatial_extras::query::{SpatialArgs, SpatialArgsParser, SpatialOperation};

const CTX_ARGS: &[&[(&str, &str)]] = &[
    &[],
    &[
        ("geo", "false"),
        ("worldBounds", "ENVELOPE(-1000, 1000, 1000, -1000)"),
    ],
    &[(
        "spatialContextFactory",
        "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory",
    )],
    &[
        (
            "spatialContextFactory",
            "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory",
        ),
        ("planetModel", "wgs84"),
    ],
];

/// `GenSpatialPrefixTree.TREES`: context, then `makeSPT` args.
const TREES: &[&[&str]] = &[
    &["0", "prefixTree", "geohash", "maxLevels", "6"],
    &["0", "prefixTree", "quad", "maxLevels", "12"],
    &["1", "prefixTree", "quad", "maxLevels", "10"],
    &["0", "prefixTree", "packedQuad", "maxLevels", "12"],
    &[
        "0",
        "prefixTree",
        "packedQuad",
        "maxLevels",
        "10",
        "noprune",
        "1",
    ],
    &["2", "prefixTree", "s2", "maxLevels", "5"],
    &["3", "prefixTree", "s2", "maxLevels", "4"],
    &["2", "prefixTree", "quad", "maxLevels", "8"],
    &["0"],
    &["1", "maxDistErr", "0.5"],
    &["0", "prefixTree", "packedQuad"],
    &["2", "prefixTree", "s2", "maxDistErr", "0.01"],
    &["2", "s2arity", "2", "maxLevels", "6"],
    &["3", "s2arity", "3", "maxLevels", "3"],
];

struct Env {
    ctx: Vec<Arc<SpatialContext>>,
    grid: Vec<Arc<dyn SpatialPrefixTree>>,
    grid_ctx: Vec<usize>,
    drt: Vec<DateRangePrefixTree>,
}

fn env() -> Env {
    let ctx: Vec<Arc<SpatialContext>> = CTX_ARGS
        .iter()
        .map(|a| {
            let m: BTreeMap<String, String> = a
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            SpatialContextFactory::make_spatial_context(&m).unwrap()
        })
        .collect();
    let mut grid: Vec<Arc<dyn SpatialPrefixTree>> = Vec::new();
    let mut grid_ctx = Vec::new();
    for t in TREES {
        let c: usize = t[0].parse().unwrap();
        let mut m = BTreeMap::new();
        let mut noprune = false;
        let mut arity = 0;
        for kv in t[1..].chunks(2) {
            if kv[0] == "noprune" {
                noprune = true;
            } else if kv[0] == "s2arity" {
                arity = kv[1].parse().unwrap();
            } else {
                m.insert(kv[0].to_string(), kv[1].to_string());
            }
        }
        let g: Arc<dyn SpatialPrefixTree> = if arity > 0 {
            let levels = m["maxLevels"].parse().unwrap();
            Arc::new(S2PrefixTree::new(ctx[c].clone(), levels, arity).unwrap())
        } else {
            make_spt(&m, &ctx[c]).unwrap()
        };
        if noprune {
            g.as_any()
                .downcast_ref::<PackedQuadPrefixTree>()
                .unwrap()
                .set_prune_leafy_branches(false);
        }
        grid.push(g);
        grid_ctx.push(c);
    }
    let drt = vec![
        DateRangePrefixTree::new(&Calendar::new_default()).unwrap(),
        DateRangePrefixTree::new(&Calendar::new_proleptic()).unwrap(),
    ];
    Env {
        ctx,
        grid,
        grid_ctx,
        drt,
    }
}

fn h(v: f64) -> String {
    format!("{:x}", v.to_bits())
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

fn unesc(s: &str) -> String {
    s.replace("\\t", "\t")
        .replace("\\n", "\n")
        .replace("\\\\", "\\")
}

fn err(e: &Error) -> String {
    let mut s = format!("ERR {} {}", e.java_class(), esc(&e.to_string()));
    if let Error::Parse { offset, .. } = e {
        s.push_str(&format!(" @{offset}"));
    }
    s
}

fn safe<T: ToString>(r: Result<T, Error>) -> String {
    match r {
        Ok(v) => v.to_string(),
        Err(e) => err(&e),
    }
}

fn hex(b: &[u8]) -> String {
    if b.is_empty() {
        return "-".into();
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn rel_name(c: &dyn Cell) -> String {
    c.shape_rel()
        .map_or("null".to_string(), |r| r.name().to_string())
}

fn cell(c: &dyn Cell) -> String {
    format!(
        "{}/{}/{}/{}",
        hex(&c.token_bytes_with_leaf()),
        c.level(),
        if c.is_leaf() { "L" } else { "-" },
        rel_name(c)
    )
}

/// `GenSpatialPrefixTree.cells(it, max)`.
fn cells(mut it: Box<dyn CellIterator>, max: usize) -> Result<String, Error> {
    let mut sb = String::new();
    let mut hash: i64 = 0;
    let mut n = 0;
    while it.has_next()? {
        let c = it.next()?;
        let s = cell(&*c);
        for ch in s.encode_utf16() {
            hash = hash.wrapping_mul(31).wrapping_add(ch as i64);
        }
        if n < 12 {
            sb.push(' ');
            sb.push_str(&s);
        }
        n += 1;
        if n >= max {
            sb.push_str(" TRUNC");
            break;
        }
    }
    Ok(format!("{n} {:x}{sb}", hash as u64))
}

fn bbox(s: &dyn Shape) -> Result<String, Error> {
    let r = s.bounding_box()?;
    Ok(format!(
        "{},{},{},{}",
        h(r.min_x()),
        h(r.max_x()),
        h(r.min_y()),
        h(r.max_y())
    ))
}

fn raw(u: &UnitNRShape) -> String {
    format!(
        "[{}]",
        u.vals()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Java's `Calendar` field constants, for `getTreeLevelForCalendarField`.
const CAL_FIELDS: [i32; 10] = [0, 1, 2, 3, 5, 11, 12, 13, 14, 6];

fn compute(env: &Env, op: &str, f: &[&str]) -> String {
    match op {
        "tree" => {
            let g = &env.grid[f[0].parse::<usize>().unwrap()];
            let mut sb = format!("{} | {} |", esc(&g.to_string()), g.max_levels());
            for d in [
                0.0, 1e-6, 1e-3, 0.01, 0.1, 1.0, 5.0, 20.0, 90.0, 200.0, 1000.0,
            ] {
                sb.push_str(&format!(" {}", g.level_for_distance(d)));
            }
            sb.push_str(" |");
            for l in 0..=g.max_levels() + 1 {
                sb.push_str(&format!(" {}", safe(g.distance_for_level(l).map(h))));
            }
            sb
        }
        "cells" => {
            let t: usize = f[0].parse().unwrap();
            let (g, c) = (&env.grid[t], env.grid_ctx[t]);
            let wkt = unesc(f[1]);
            let pct: f64 = f[2].parse().unwrap();
            let dl: i32 = f[3].parse().unwrap();
            safe((|| {
                let s = env.ctx[c].read_shape_from_wkt(&wkt)?;
                let detail = if pct == 0.0 {
                    dl
                } else {
                    g.level_for_distance(SpatialArgs::calc_distance_from_err_pct(
                        &*s,
                        pct,
                        &env.ctx[c],
                    )?)
                };
                Ok(format!(
                    "{detail} {}",
                    cells(g.tree_cell_iterator(&s, detail)?, 3000)?
                ))
            })())
        }
        "read" => {
            let t: usize = f[0].parse().unwrap();
            let (g, c) = (&env.grid[t], env.grid_ctx[t]);
            let wkt = unesc(f[3]);
            safe((|| {
                let cl = g.read_cell(&unhex(f[1]))?;
                let o = g.read_cell(&unhex(f[2]))?;
                let mut sb = cell(&*cl);
                sb.push_str(&format!(" | {}", safe(cl.shape().and_then(|s| bbox(&*s)))));
                sb.push_str(&format!(
                    " | {} {}",
                    cl.is_prefix_of(&*o),
                    cl.compare_to_no_leaf(&*o).signum()
                ));
                let s = env.ctx[c].read_shape_from_wkt(&wkt)?;
                sb.push_str(&format!(
                    " | {}",
                    safe(cl.shape().and_then(|cs| cs.relate(&*s)).map(|r| r.name()))
                ));
                if cl.level() < g.max_levels() {
                    sb.push_str(&format!(
                        " | {}",
                        safe(cl.next_level_cells(Some(&s)).and_then(|it| cells(it, 100)))
                    ));
                    sb.push_str(&format!(
                        " | {}",
                        safe(cl.next_level_cells(None).and_then(|it| cells(it, 100)))
                    ));
                }
                Ok(sb)
            })())
        }
        "drt" => {
            let tree = &env.drt[f[0].parse::<usize>().unwrap()];
            let min = tree.to_unit_shape_millis(i64::MIN);
            let max = tree.to_unit_shape_millis(i64::MAX);
            let mut sb = format!(
                "{} {} {} {} {} {}",
                tree,
                tree.max_levels(),
                raw(&min),
                raw(&max),
                esc(&min.to_string()),
                esc(&max.to_string())
            );
            for fld in CAL_FIELDS {
                sb.push_str(&format!(
                    " {}",
                    safe(tree.tree_level_for_calendar_field(fld))
                ));
            }
            sb
        }
        "date" => {
            let tree = &env.drt[f[0].parse::<usize>().unwrap()];
            let s = unesc(f[1]);
            safe((|| {
                let mut cal = tree.parse_calendar(&s)?;
                let st = tree.cal_to_string(&mut cal);
                let u = tree.to_shape(&mut cal);
                let back = tree.cal_to_string(&mut tree.to_calendar(&u));
                Ok(format!(
                    "{} {} {} {} {} {} {}",
                    esc(&st),
                    raw(&u),
                    hex(&u.token_bytes_no_leaf()),
                    esc(&u.to_string()),
                    esc(&back),
                    DateRangePrefixTree::cal_precision_field(&cal),
                    tree.to_calendar(&u).time_in_millis()
                ))
            })())
        }
        "datems" => {
            let tree = &env.drt[f[0].parse::<usize>().unwrap()];
            let ms: i64 = f[1].parse().unwrap();
            let level: i32 = f[2].parse().unwrap();
            let u = tree.to_unit_shape_millis(ms);
            let r = u.round_to_level(level);
            let mut sb = format!(
                "{} {} {} {} {}",
                raw(&u),
                esc(&u.to_string()),
                raw(&r),
                esc(&r.to_string()),
                hex(&r.token_bytes_no_leaf())
            );
            for l in 0..r.level() {
                sb.push_str(if l == 0 { " " } else { "," });
                sb.push_str(&safe(tree.num_sub_cells(&r.shape_at_level(l))));
            }
            sb
        }
        "subcells" => {
            let tree = &env.drt[f[0].parse::<usize>().unwrap()];
            let year: i32 = f[1].parse().unwrap();
            let month: i32 = f[2].parse().unwrap();
            let mut cal = tree.new_cal();
            cal.set(ERA, if year <= 0 { 0 } else { 1 });
            cal.set(YEAR, if year <= 0 { 1 - year } else { year });
            cal.set(MONTH, month);
            let u = tree.to_shape(&mut cal);
            format!(
                "{} {} {}",
                raw(&u),
                safe(tree.num_sub_cells(&u)),
                esc(&u.to_string())
            )
        }
        "nrshape" => {
            let tree = &env.drt[f[0].parse::<usize>().unwrap()];
            let (s, o) = (unesc(f[1]), unesc(f[2]));
            let lvl: i32 = f[3].parse().unwrap();
            safe((|| {
                let a = tree.parse_shape(&s)?;
                let mut sb = esc(&a.to_string());
                sb.push_str(&format!(
                    " | {}",
                    safe(a.round_to_level(lvl).map(|r| esc(&r.to_string())))
                ));
                let a_shape = a.clone().into_shape();
                sb.push_str(&format!(
                    " | {}",
                    safe(
                        tree.parse_shape(&o)
                            .and_then(|b| a_shape.relate(&*b.into_shape()))
                            .map(|r| r.name())
                    )
                ));
                sb.push_str(&format!(
                    " | {}",
                    safe(
                        tree.parse_shape(&o)
                            .map(|b| a_shape.equals(&*b.into_shape()))
                    )
                ));
                sb.push_str(&format!(
                    " | {}",
                    safe(
                        tree.tree_cell_iterator(&a_shape, tree.max_levels())
                            .and_then(|it| cells(it, 2000))
                    )
                ));
                sb.push_str(&format!(
                    " | {}",
                    safe(
                        tree.tree_cell_iterator(&a_shape, lvl)
                            .and_then(|it| cells(it, 2000))
                    )
                ));
                Ok(sb)
            })())
        }
        "nrrel" => {
            let tree = &env.drt[f[0].parse::<usize>().unwrap()];
            let (a, b) = (unesc(f[1]), unesc(f[2]));
            safe((|| {
                let (x, y) = (tree.parse_shape(&a)?, tree.parse_shape(&b)?);
                let cmp = match (&x, &y) {
                    (NRShape::Unit(ux), NRShape::Unit(uy)) => (ux.compare_to(uy) as i8).to_string(),
                    _ => "-".to_string(),
                };
                let rel = x.into_shape().relate(&*y.into_shape())?;
                Ok(format!("{} {cmp}", rel.name()))
            })())
        }
        "nrread" => {
            let tree = &env.drt[f[0].parse::<usize>().unwrap()];
            let ms: i64 = f[1].parse().unwrap();
            let lvl: i32 = f[2].parse().unwrap();
            let leaf = f[3] == "true";
            let u = tree.to_unit_shape_millis(ms).round_to_level(lvl);
            let mut term = u.token_bytes_no_leaf();
            if leaf {
                term.push(0);
            }
            safe((|| {
                let back = tree.read_cell(&term)?;
                Ok(format!(
                    "{} {} {} {}",
                    hex(&term),
                    cell(&*back),
                    esc(&back.shape()?.to_string()),
                    safe(back.next_level_cells(None).and_then(|it| cells(it, 50)))
                ))
            })())
        }
        "args" => {
            let c: usize = f[0].parse().unwrap();
            let ctx = &env.ctx[c];
            let a = unesc(f[1]);
            safe(SpatialArgsParser.parse(&a, ctx).map(|args| {
                let opt = |v: Option<f64>| v.map_or("null".to_string(), |x| x.to_string());
                format!(
                    "{} | {} | {} {}",
                    if c >= 2 {
                        args.operation.name().to_string()
                    } else {
                        esc(&args.to_string())
                    },
                    safe(args.resolve_dist_err(ctx, 0.025).map(h)),
                    opt(args.dist_err_pct()),
                    opt(args.dist_err())
                )
            }))
        }
        "op" => {
            let c: usize = f[0].parse().unwrap();
            let ctx = &env.ctx[c];
            safe((|| {
                let sa = ctx.read_shape_from_wkt(&unesc(f[1]))?;
                let sb = ctx.read_shape_from_wkt(&unesc(f[2]))?;
                let mut out = Vec::new();
                for op in SpatialOperation::VALUES {
                    out.push(format!("{}={}", op.name(), safe(op.evaluate(&*sa, &*sb))));
                }
                Ok(out.join(" "))
            })())
        }
        other => panic!("unknown op {other}"),
    }
}

/// Java prints `Double`'s `toString` for the args' error fields; Rust's
/// `f64` display differs for some values, so those fields are compared as
/// numbers by [`normalise`].
fn normalise(op: &str, text: &str) -> String {
    let mut t = text.to_string();
    if op == "args" {
        if let Some(i) = t.rfind(" | ") {
            let tail: Vec<String> = t[i + 3..]
                .split(' ')
                .map(|v| {
                    v.parse::<f64>()
                        .map_or(v.to_string(), |x| format!("{:x}", x.to_bits()))
                })
                .collect();
            t = format!("{} | {}", &t[..i], tail.join(" "));
        }
    }
    // A NaN's sign differs between x86-64 and arm64.
    let bytes = t.as_bytes();
    let mut out = String::with_capacity(t.len());
    let mut i = 0;
    while i < bytes.len() {
        let run = bytes[i..]
            .iter()
            .take_while(|b| b.is_ascii_hexdigit())
            .count();
        if run == 0 {
            let skip = bytes[i..]
                .iter()
                .position(u8::is_ascii_hexdigit)
                .unwrap_or(bytes.len() - i);
            out.push_str(&t[i..i + skip]);
            i += skip;
            continue;
        }
        let token = &t[i..i + run];
        let is_nan =
            run == 16 && u64::from_str_radix(token, 16).is_ok_and(|b| f64::from_bits(b).is_nan());
        out.push_str(if is_nan { "NaN" } else { token });
        i += run;
    }
    out
}

#[test]
fn prefix_trees_match_lucene() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/spatial_prefix_tree/trees.tsv");
    let text = std::fs::read_to_string(path)
        .expect("run scripts/gen-fixtures.sh --only GenSpatialPrefixTree");
    let env = env();
    let mut failures = Vec::new();
    let mut checked = 0;
    for line in text.lines() {
        let (lhs, expected) = line.split_once("\t=>\t").expect("a record");
        let mut fields: Vec<&str> = lhs.split('\t').collect();
        let op = fields.remove(0);
        let actual = compute(&env, op, &fields);
        checked += 1;
        if normalise(op, &actual) != normalise(op, expected) {
            failures.push(format!("{lhs}\n   java: {expected}\n   rust: {actual}"));
        }
    }
    assert!(checked > 3000, "only {checked} records");
    assert!(
        failures.is_empty(),
        "{} of {checked} records differ:\n{}",
        failures.len(),
        failures
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn calendar_field_constants_match_java() {
    // `Calendar`'s field numbers, as `DateRangePrefixTree` names them.
    assert_eq!(
        [
            ERA,
            YEAR,
            MONTH,
            DAY_OF_MONTH,
            HOUR_OF_DAY,
            MINUTE,
            SECOND,
            MILLISECOND
        ],
        [0, 1, 2, 5, 11, 12, 13, 14]
    );
}
