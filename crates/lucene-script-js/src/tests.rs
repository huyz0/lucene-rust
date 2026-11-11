use super::*;

/// Fields served from memory: per field, per document of the call, its
/// values (a keyword's ordinals into `terms`).
struct Mem {
    docs: Vec<Vec<Vec<f64>>>,
    terms: Vec<String>,
    loads: usize,
}

impl Fields for Mem {
    fn load(
        &mut self,
        field: usize,
        values: &mut Vec<f64>,
        starts: &mut Vec<i32>,
    ) -> Result<(), String> {
        self.loads += 1;
        for doc in &self.docs[field] {
            starts.push(values.len() as i32);
            values.extend_from_slice(doc);
        }
        starts.push(values.len() as i32);
        Ok(())
    }

    fn term(&mut self, _field: usize, ord: i64) -> Result<String, String> {
        self.terms
            .get(ord as usize)
            .cloned()
            .ok_or_else(|| format!("no term {ord}"))
    }
}

fn mem(docs: Vec<Vec<Vec<f64>>>) -> Mem {
    Mem {
        docs,
        terms: vec!["a".into(), "b".into(), "c".into()],
        loads: 0,
    }
}

fn script(source: &str, fields: &[(&str, Kind)]) -> Script {
    Script::new(
        source,
        fields.iter().map(|(n, k)| (n.to_string(), *k)).collect(),
        Limits::default(),
    )
}

fn params(entries: &[(&str, Param)]) -> Params {
    Params::new(&Param::Map(
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
}

fn batch(
    s: &Script,
    p: &Params,
    f: &mut Mem,
    count: usize,
    scores: Option<&[f32]>,
) -> Result<Vec<f64>, Failure> {
    let mut out = vec![0.0; count];
    run_batch(s, p, 0, count, scores, false, f, &mut out)?;
    Ok(out)
}

#[test]
fn an_expression_scores_each_document() {
    let s = script(
        "doc['likes'].value * params.w + _score",
        &[("likes", Kind::Number)],
    );
    let p = params(&[("w", Param::Number(2.0))]);
    let mut f = mem(vec![vec![vec![1.0], vec![2.0], vec![3.5]]]);
    let out = batch(&s, &p, &mut f, 3, Some(&[0.5, 1.0, 0.0])).unwrap();
    assert_eq!(out, vec![2.5, 5.0, 7.0]);
    // One load for the whole call.
    assert_eq!(f.loads, 1);
}

#[test]
fn a_body_returns_its_value() {
    let s = script(
        "let t = 0;\nfor (const v of doc['n']) t += v;\nreturn t / doc['n'].size();",
        &[("n", Kind::Number)],
    );
    let mut f = mem(vec![vec![vec![1.0, 2.0, 6.0], vec![4.0]]]);
    assert_eq!(
        batch(&s, &params(&[]), &mut f, 2, None).unwrap(),
        vec![3.0, 4.0]
    );
}

#[test]
fn a_filter_returns_booleans() {
    let s = script("doc['ok'].value", &[("ok", Kind::Boolean)]);
    let mut f = mem(vec![vec![vec![1.0], vec![0.0]]]);
    let mut out = vec![9.0; 2];
    run_batch(&s, &params(&[]), 0, 2, None, true, &mut f, &mut out).unwrap();
    assert_eq!(out, vec![1.0, 0.0]);
    let s = script("1", &[]);
    let e = run_batch(&s, &params(&[]), 0, 1, None, true, &mut f, &mut out).unwrap_err();
    assert!(
        matches!(&e, Failure::Thrown { name, message, line: None, .. } if name == "TypeError" && message.contains("must return a boolean, not number")),
        "{e:?}"
    );
}

#[test]
fn a_score_must_be_a_number() {
    let s = script("'x'", &[]);
    let e = batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap_err();
    assert!(
        e.to_string().contains("must return a number, not string"),
        "{e}"
    );
}

#[test]
fn keywords_read_their_terms() {
    let s = script(
        "doc['tag'].value + ':' + doc['tag'].size() + ':' + doc['tag'].values.join('|')",
        &[("tag", Kind::Keyword)],
    );
    let mut f = mem(vec![vec![vec![2.0, 0.0]]]);
    let out = run_one(&s, &params(&[]), 7, None, &mut f).unwrap();
    assert_eq!(out, Out::String("c:2:c|a".into()));
}

#[test]
fn one_converts_what_the_script_returns() {
    let mut f = mem(vec![]);
    let p = params(&[]);
    let run = |src: &str, f: &mut Mem| run_one(&script(src, &[]), &p, 0, Some(1.5), f).unwrap();
    assert_eq!(run("null", &mut f), Out::Null);
    assert_eq!(run("undefined", &mut f), Out::Null);
    assert_eq!(run("_score", &mut f), Out::Number(1.5));
    assert_eq!(run("true", &mut f), Out::Bool(true));
    assert_eq!(
        run("[1, 'x', [false]]", &mut f),
        Out::List(vec![
            Out::Number(1.0),
            Out::String("x".into()),
            Out::List(vec![Out::Bool(false)])
        ])
    );
    assert_eq!(run("({})", &mut f), Out::Other("object".into()));
    assert_eq!(run("10n", &mut f), Out::Other("bigint".into()));
}

#[test]
fn a_document_without_values_throws_as_painless_does() {
    let s = script("doc['n'].value", &[("n", Kind::Number)]);
    let mut f = mem(vec![vec![vec![1.0], vec![]]]);
    let e = batch(&s, &params(&[]), &mut f, 2, None).unwrap_err();
    match e {
        Failure::Thrown {
            at, name, message, ..
        } => {
            assert_eq!(at, 1);
            assert_eq!(name, "Error");
            assert!(message.starts_with("A document doesn't have a value for a field!"));
        }
        e => panic!("{e:?}"),
    }
    let s = script("doc['n'].get(3)", &[("n", Kind::Number)]);
    let mut f = mem(vec![vec![vec![1.0]]]);
    let e = batch(&s, &params(&[]), &mut f, 1, None).unwrap_err();
    assert!(
        e.to_string().contains("Index 3 out of bounds for length 1"),
        "{e}"
    );
}

#[test]
fn unknown_and_unreadable_fields_fail_when_read() {
    let s = script("doc['nope'].value", &[]);
    let e = batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap_err();
    assert!(
        e.to_string()
            .contains("No field found for [nope] in mapping"),
        "{e}"
    );
    let s = script("doc.containsKey('t') ? 1 : 0", &[("t", Kind::Unreadable)]);
    assert_eq!(
        batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap(),
        vec![1.0]
    );
    let s = script("doc['t'].value", &[("t", Kind::Unreadable)]);
    let e = batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap_err();
    assert!(
        e.to_string().contains("no doc values a js script can read"),
        "{e}"
    );
}

#[test]
fn an_error_names_its_line() {
    let s = script("let x = 1;\nnull.y;\nreturn x;", &[]);
    let e = batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap_err();
    match e {
        Failure::Thrown {
            name, line, column, ..
        } => {
            assert_eq!(name, "TypeError");
            assert_eq!(line, Some(2));
            assert_eq!(column, Some(5));
        }
        e => panic!("{e:?}"),
    }
    let s = script("throw 'plain'", &[]);
    let e = batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap_err();
    assert!(
        matches!(&e, Failure::Thrown { name, message, .. } if name == "Thrown" && message == "plain"),
        "{e:?}"
    );
}

#[test]
fn a_syntax_error_fails_the_check() {
    let e = script("return (", &[]).check().unwrap_err();
    match &e {
        Failure::Compile { message, line, .. } => {
            assert!(message.starts_with("SyntaxError"), "{message}");
            assert_eq!(*line, Some(1));
        }
        e => panic!("{e:?}"),
    }
    assert!(
        e.to_string().starts_with("compile error: SyntaxError"),
        "{e}"
    );
    script("1 + 1", &[]).check().unwrap();
    // A body cannot close the function it is compiled into.
    assert!(script("1); }); (function () { (0", &[]).check().is_err());
}

#[test]
fn scripts_cannot_reach_out_or_keep_state() {
    let mut f = mem(vec![]);
    let p = params(&[("m", Param::Map(vec![("k".into(), Param::Number(1.0))]))]);
    for (src, want) in [
        ("eval('1')", "EvalError"),
        ("new Function('return 1')()", "EvalError"),
        (
            "typeof WebAssembly === 'undefined' ? null.x : 1",
            "TypeError",
        ),
        ("globalThis.leak = 1; return 1", "TypeError"),
        ("Array.prototype.push = null; return 1", "TypeError"),
        ("params.m.k = 2; return 1", "TypeError"),
        ("undeclared = 1; return 1", "ReferenceError"),
        // Below the globals too: their functions, namespaces and the
        // prototypes no global names.
        ("Math.max.n = 1; return 1", "TypeError"),
        ("JSON.parse.x = 1; return 1", "TypeError"),
        (
            "Object.getPrototypeOf(function* () {}).x = 1; return 1",
            "TypeError",
        ),
        (
            "Object.getPrototypeOf([].values().map((x) => x)).x = 1; return 1",
            "TypeError",
        ),
        (
            "Object.getPrototypeOf(/a/[Symbol.matchAll]('a')).x = 1; return 1",
            "TypeError",
        ),
        // No memory outside the heap V8 limits.
        ("new Uint8Array(1 << 30)", "ReferenceError"),
        ("new ArrayBuffer(8)", "ReferenceError"),
        ("new SharedArrayBuffer(8)", "ReferenceError"),
        ("new DataView(null)", "ReferenceError"),
        ("Intl.Collator", "ReferenceError"),
    ] {
        let e = batch(&script(src, &[]), &p, &mut f, 1, None).unwrap_err();
        assert!(
            matches!(&e, Failure::Thrown { name, .. } if name == want),
            "{src}: {e:?}"
        );
    }
    for src in [
        "typeof require",
        "typeof setTimeout",
        "typeof process",
        "typeof fetch",
    ] {
        assert_eq!(
            run_one(&script(src, &[]), &p, 0, None, &mut f).unwrap(),
            Out::String("undefined".into()),
            "{src}"
        );
    }
}

#[test]
fn deep_params_freeze_all_the_way_down() {
    let mut deep = Param::Map(vec![]);
    for _ in 0..200 {
        deep = Param::Map(vec![("a".into(), deep)]);
    }
    let p = params(&[("d", deep)]);
    let s = script(
        "let o = params.d; while (o.a) o = o.a; o.x = 1; return 1",
        &[],
    );
    let e = batch(&s, &p, &mut mem(vec![]), 1, None).unwrap_err();
    assert!(
        matches!(&e, Failure::Thrown { name, .. } if name == "TypeError"),
        "{e:?}"
    );
}

#[test]
fn a_field_is_made_only_by_the_runner() {
    let mut f = mem(vec![vec![vec![1.0]], vec![vec![2.0]]]);
    let s = script(
        "new (doc['a'].constructor)({}, 1).value",
        &[("a", Kind::Number), ("b", Kind::Unreadable)],
    );
    let e = batch(&s, &params(&[]), &mut f, 1, None).unwrap_err();
    assert!(
        matches!(&e, Failure::Thrown { name, message, .. } if name == "TypeError" && message == "Illegal constructor"),
        "{e:?}"
    );
}

#[test]
fn a_field_read_first_runs_its_document_again() {
    // `b` is read on the second document only, inside a try: the script
    // never sees the read that loads it.
    let s = script(
        "if (doc['a'].value < 2) return 0; try { return doc['b'].value; } catch (e) { return -1; }",
        &[("a", Kind::Number), ("b", Kind::Number)],
    );
    let p = params(&[]);
    let mut f = mem(vec![
        vec![vec![1.0], vec![2.0], vec![3.0]],
        vec![vec![10.0], vec![20.0], vec![30.0]],
    ]);
    assert_eq!(
        batch(&s, &p, &mut f, 3, None).unwrap(),
        vec![0.0, 20.0, 30.0]
    );
    assert_eq!(f.loads, 2);
    // Later calls load both up front.
    f.loads = 0;
    assert_eq!(
        batch(&s, &p, &mut f, 3, None).unwrap(),
        vec![0.0, 20.0, 30.0]
    );
    assert_eq!(f.loads, 2);
    let mut one = mem(vec![vec![vec![5.0]], vec![vec![7.0]]]);
    assert_eq!(
        run_one(&s, &p, 0, None, &mut one).unwrap(),
        Out::Number(7.0)
    );
}

#[test]
fn a_throw_keeps_the_documents_before_it() {
    let s = script(
        "if (doc['a'].value === 3) throw new Error('three'); return doc['a'].value * 2",
        &[("a", Kind::Number)],
    );
    let mut f = mem(vec![vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]]);
    let mut out = vec![0.0; 4];
    let e = run_batch(&s, &params(&[]), 0, 4, None, false, &mut f, &mut out).unwrap_err();
    assert!(
        matches!(&e, Failure::Thrown { at: 2, message, .. } if message == "three"),
        "{e:?}"
    );
    assert_eq!(&out[..2], &[2.0, 4.0]);
}

#[test]
fn one_large_allocation_stops_the_script_and_not_the_process() {
    std::thread::spawn(|| {
        let limits = Limits {
            timeout: Duration::from_secs(60),
            max_heap: 16 << 20,
        };
        for src in [
            // Small objects, which V8's heap limit stops.
            "const a = []; for (let i = 0; i < 64; i++) a.push(new Array(1 << 20).fill(i)); return a.length",
            // Large strings, which it lets past: the watchdog stops them.
            "const a = []; for (let i = 0; ; i++) { const s = 'x'.repeat(1 << 23) + i; s.indexOf('y'); a.push(s); }",
            // One string far past the limit at once.
            "const s = 'x'.repeat(1 << 28) + 'y'; s.indexOf('z'); for (;;) {}",
        ] {
            let s = Script::new(src, vec![], limits);
            let e = batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap_err();
            assert_eq!(e, Failure::OutOfMemory(16 << 20), "{src}");
        }
    })
    .join()
    .unwrap();
}

#[test]
fn a_search_stops_at_its_deadline() {
    let s = Script::new(
        "while (true) {}",
        vec![],
        Limits {
            timeout: Duration::from_millis(50),
            ..Limits::default()
        },
    );
    let t = Instant::now();
    let e = batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap_err();
    assert_eq!(e, Failure::TimedOut(Duration::from_millis(50)));
    assert!(t.elapsed() < Duration::from_secs(5));
    // The deadline holds for the rest of the search, and the isolate runs on.
    assert_eq!(
        batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap_err(),
        e
    );
    assert_eq!(
        batch(&script("3", &[]), &params(&[]), &mut mem(vec![]), 1, None).unwrap(),
        vec![3.0]
    );
}

#[test]
fn a_heap_past_its_limit_stops_the_script() {
    std::thread::spawn(|| {
        let s = Script::new(
            "const a = []; while (true) a.push({ x: [1, 2, 3, 4] });",
            vec![],
            Limits {
                timeout: Duration::from_secs(60),
                max_heap: 16 << 20,
            },
        );
        let e = batch(&s, &params(&[]), &mut mem(vec![]), 1, None).unwrap_err();
        assert_eq!(e, Failure::OutOfMemory(16 << 20));
        // A new isolate runs the next script.
        assert_eq!(
            batch(&script("4", &[]), &params(&[]), &mut mem(vec![]), 1, None).unwrap(),
            vec![4.0]
        );
    })
    .join()
    .unwrap();
}

#[test]
fn a_host_failure_is_reported_as_the_hosts() {
    let mut f = mem(vec![vec![]]);
    let s = script("doc['x'].value", &[("x", Kind::Number)]);
    // Field 0 in a call of 2 documents with none loaded: a short column.
    let e = batch(&s, &params(&[]), &mut f, 2, None).unwrap_err();
    assert!(
        matches!(&e, Failure::Host(m) if m.contains("1 starts for 2 documents")),
        "{e:?}"
    );
}

#[test]
fn columns_are_reused_and_regrown() {
    let s = script("doc['n'].value", &[("n", Kind::Number)]);
    let p = params(&[]);
    for n in [1usize, 40, 3, 200] {
        let mut f = mem(vec![(0..n).map(|i| vec![i as f64]).collect()]);
        let out = batch(&s, &p, &mut f, n, None).unwrap();
        assert_eq!(out, (0..n).map(|i| i as f64).collect::<Vec<_>>());
    }
}

#[test]
fn params_convert_to_json() {
    let p = Param::Map(vec![
        ("s".into(), Param::String("q\"\\\n\u{2028}".into())),
        (
            "l".into(),
            Param::List(vec![
                Param::Null,
                Param::Bool(false),
                Param::Number(f64::NAN),
            ]),
        ),
    ]);
    let mut json = String::new();
    p.json(&mut json);
    assert_eq!(
        json,
        "{\"s\":\"q\\\"\\\\\\u000a\\u2028\",\"l\":[null,false,null]}"
    );
    let s = script("params.s.length + params.l.length", &[]);
    assert_eq!(
        batch(&s, &Params::new(&p), &mut mem(vec![]), 1, None).unwrap(),
        vec![8.0]
    );
}

#[test]
fn many_scripts_and_params_stay_bounded() {
    let mut f = mem(vec![]);
    for i in 0..(MAX_SCRIPTS + 3) {
        let s = script(&format!("{i}"), &[]);
        let p = params(&[("i", Param::Number(i as f64))]);
        assert_eq!(batch(&s, &p, &mut f, 1, None).unwrap(), vec![i as f64]);
    }
    let s = script("params.i", &[]);
    for i in 0..(MAX_PARAMS + 3) {
        let p = params(&[("i", Param::Number(i as f64))]);
        assert_eq!(batch(&s, &p, &mut f, 1, None).unwrap(), vec![i as f64]);
    }
    ENGINE.with(|e| {
        let e = e.borrow();
        let e = e.as_ref().unwrap();
        assert!(e.scripts.len() <= MAX_SCRIPTS);
        assert!(e.scripts.values().all(|c| c.params.len() <= MAX_PARAMS));
    });
}

#[test]
fn failures_print_where_they_happened() {
    let e = Failure::Thrown {
        at: 0,
        name: "TypeError".into(),
        message: "m".into(),
        line: Some(3),
        column: Some(0),
    };
    assert_eq!(e.to_string(), "TypeError: m (line 3, column 1)");
    let e = Failure::Compile {
        message: "SyntaxError: x".into(),
        line: Some(2),
        column: None,
    };
    assert_eq!(e.to_string(), "compile error: SyntaxError: x (line 2)");
    assert_eq!(
        Failure::OutOfMemory(64 << 20).to_string(),
        "the js script ran past its heap limit of [64mb]"
    );
    assert_eq!(
        Failure::TimedOut(Duration::from_millis(5)).to_string(),
        "the js script ran past its time limit of [5ms]"
    );
    assert_eq!(Failure::Host("io".into()).to_string(), "io");
}

/// ns per document, batched against one call per document:
/// `cargo test --release --manifest-path crates/lucene-script-js/Cargo.toml -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_batch_against_per_document() {
    const DOCS: usize = 1 << 20;
    const BATCH: usize = 1024;
    let s = script(
        "doc['likes'].value * params.w + _score",
        &[("likes", Kind::Number)],
    );
    let p = params(&[("w", Param::Number(2.5))]);
    let mut f = mem(vec![(0..BATCH).map(|i| vec![(i % 1000) as f64]).collect()]);
    let scores = vec![1.5f32; BATCH];
    let mut out = vec![0.0; BATCH];
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..DOCS / BATCH {
            run_batch(&s, &p, 0, BATCH, Some(&scores), false, &mut f, &mut out).unwrap();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / DOCS as f64);
    }
    println!("batch of {BATCH}: {best:.1} ns/doc");
    let mut one = mem(vec![vec![vec![7.0]]]);
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..DOCS / 16 {
            run_one(&s, &p, 0, Some(1.5), &mut one).unwrap();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / (DOCS / 16) as f64);
    }
    println!("one call per document: {best:.1} ns/doc");
    let s = script("1", &[]);
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..DOCS / 16 {
            run_one(&s, &p, 0, None, &mut one).unwrap();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / (DOCS / 16) as f64);
    }
    println!("one call per document, no field: {best:.1} ns/doc");
}
