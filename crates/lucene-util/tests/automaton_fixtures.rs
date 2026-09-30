//! Lucene 10.5.0's automaton API, differentially: `fixtures/src/GenAutomata.java`
//! records what `RegExp`, `LevenshteinAutomata`, `Operations`, `Automata`,
//! `CompiledAutomaton` and the run automata produce for a corpus of inputs,
//! and this test recomputes every recorded line with `lucene_util::automaton`
//! and compares it verbatim.
//!
//! Each fixture case is a `CASE` line (the construction and its inputs), a
//! `probes` line (extra probe strings) and result lines. The comparison
//! covers raw and determinized state/transition counts *and full transition
//! tables* (so Lucene's state numbering, not just the language), the
//! `TooComplexToDeterminizeException` message at the default work limit,
//! minimized counts, probe acceptance through four different runners,
//! common prefix, singleton, topological order, finite strings, and the
//! `CompiledAutomaton` type/term/suffix/sink/floor.

use std::path::PathBuf;

use lucene_util::automaton::automata;
use lucene_util::automaton::operations as ops;
use lucene_util::automaton::{
    Automaton, AutomatonError, AutomatonType, ByteRunnable, CharacterRunAutomaton,
    CompiledAutomaton, LevenshteinAutomata, LimitedFiniteStringsIterator, NfaRunAutomaton, RegExp,
    Transition, TransitionAccessor, DEFAULT_DETERMINIZE_WORK_LIMIT,
};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/automata/automata.tsv")
}

// ---- string encoding ---------------------------------------------------------

fn esc_cps(cps: &[i32]) -> String {
    let mut b = String::new();
    for &c in cps {
        if (0x20..0x7f).contains(&c) && c != '\\' as i32 {
            b.push(c as u8 as char);
        } else {
            b.push_str(&format!("\\u{{{:x}}}", c));
        }
    }
    b
}

fn esc(s: &str) -> String {
    esc_cps(&s.chars().map(|c| c as i32).collect::<Vec<_>>())
}

fn unesc(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\\' {
            assert_eq!(it.next(), Some('u'));
            assert_eq!(it.next(), Some('{'));
            let mut hex = String::new();
            for h in it.by_ref() {
                if h == '}' {
                    break;
                }
                hex.push(h);
            }
            let cp = u32::from_str_radix(&hex, 16).unwrap();
            out.push(char::from_u32(cp).expect("probe strings are valid scalars"));
        } else {
            out.push(c);
        }
    }
    out
}

fn hex(b: Option<&[u8]>) -> String {
    match b {
        None => "-".into(),
        Some(b) => {
            let mut s = String::from("x");
            for x in b {
                s.push_str(&format!("{x:02x}"));
            }
            s
        }
    }
}

fn flag(b: bool) -> &'static str {
    if b {
        "1"
    } else {
        "0"
    }
}

fn labels(s: &str, binary: bool) -> Vec<i32> {
    if binary {
        s.bytes().map(i32::from).collect()
    } else {
        s.chars().map(|c| c as i32).collect()
    }
}

fn bits(probes: &[String], mut f: impl FnMut(&str) -> bool) -> String {
    probes
        .iter()
        .map(|p| if f(p) { '1' } else { '0' })
        .collect()
}

fn dump(a: &Automaton) -> String {
    let mut b = String::new();
    let mut t = Transition::new();
    for s in 0..a.get_num_states() {
        if s > 0 {
            b.push(';');
        }
        b.push(if a.is_accept(s) { 'A' } else { 'N' });
        let n = a.init_transition(s, &mut t);
        for _ in 0..n {
            a.get_next_transition(&mut t);
            b.push_str(&format!(",{}/{}/{}", t.dest, t.min, t.max));
        }
    }
    b
}

fn err_msg(e: &AutomatonError) -> String {
    let m = e.to_string();
    if m.is_empty() {
        "null".into()
    } else {
        m
    }
}

// ---- the per-automaton block (mirrors GenAutomata.block) -----------------------

struct Out(Vec<String>);

impl Out {
    fn line(&mut self, fields: &[&str]) {
        self.0.push(fields.join("\t"));
    }
}

fn nfa_run(r: &dyn ByteRunnable, labels: &[i32]) -> bool {
    let mut p = 0;
    for &c in labels {
        p = r.step(p, c);
        if p == -1 {
            return false;
        }
    }
    r.is_accept(p)
}

fn compiled(
    out: &mut Out,
    key: &str,
    a: &Automaton,
    binary: bool,
    probes: &[String],
    finite: bool,
) {
    let c = CompiledAutomaton::with_options(a, false, true, binary).unwrap();
    let mut run_bits = "-".to_string();
    let mut floors = "-".to_string();
    if c.automaton_type == AutomatonType::NORMAL {
        let r = c.get_byte_runnable().unwrap();
        run_bits = bits(probes, |p| r.run(p.as_bytes()));
        if c.run_automaton.is_some() && finite {
            floors = probes
                .iter()
                .map(|p| match c.floor(p.as_bytes()) {
                    None => "null".to_string(),
                    Some(f) => hex(Some(&f)),
                })
                .collect::<Vec<_>>()
                .join(",");
        }
    }
    let ty = format!("{:?}", c.automaton_type);
    let sink = c.sink_state.to_string();
    let nfa = c.automaton_type == AutomatonType::NORMAL && c.run_automaton.is_none();
    out.line(&[
        key,
        &ty,
        &hex(c.term.as_deref()),
        &hex(c.common_suffix_ref.as_deref()),
        &sink,
        flag(c.finite),
        flag(nfa),
        &run_bits,
    ]);
    out.line(&[&format!("{key}_floor"), &floors]);
}

fn block(out: &mut Out, a: &Automaton, binary: bool, probes: &[String]) {
    out.line(&[
        "raw",
        &a.get_num_states().to_string(),
        &a.get_total_num_transitions().to_string(),
        flag(a.is_deterministic()),
    ]);
    if a.get_num_states() <= 40 {
        out.line(&["rawdump", &dump(a)]);
    }
    let a_finite = ops::is_finite(a).unwrap();
    out.line(&[
        "flags",
        flag(ops::has_dead_states(a)),
        flag(ops::is_empty(a)),
        flag(if binary {
            ops::is_total_range(a, 0, 255)
        } else {
            ops::is_total(a)
        }),
        flag(a_finite),
    ]);
    let nfa = if binary {
        NfaRunAutomaton::with_alphabet(a.clone(), 256)
    } else {
        NfaRunAutomaton::new(a.clone())
    };
    out.line(&[
        "nfarun",
        &bits(probes, |p| nfa_run(&nfa, &labels(p, binary))),
    ]);
    compiled(out, "compiled", a, binary, probes, a_finite);
    let det = match ops::determinize(a, DEFAULT_DETERMINIZE_WORK_LIMIT) {
        Ok(d) => d,
        Err(e) => {
            out.line(&["det", "TOO_COMPLEX", &e.to_string()]);
            return;
        }
    };
    out.line(&[
        "det",
        &det.get_num_states().to_string(),
        &det.get_total_num_transitions().to_string(),
        flag(det.is_deterministic()),
    ]);
    if det.get_num_states() <= 40 {
        out.line(&["detdump", &dump(&det)]);
    }
    if det.get_num_states() <= 2000 {
        let min = ops::minimize(&det);
        out.line(&[
            "min",
            &min.get_num_states().to_string(),
            &min.get_total_num_transitions().to_string(),
        ]);
    }
    out.line(&[
        "run",
        &bits(probes, |p| ops::run_ints(&det, &labels(p, binary))),
    ]);
    if !binary {
        let cra = CharacterRunAutomaton::new(&det).unwrap();
        out.line(&["charrun", &bits(probes, |p| cra.run(p))]);
    }
    match ops::get_common_prefix(&det) {
        Ok(p) => out.line(&["prefix", &esc_cps(&p)]),
        Err(e) => out.line(&["prefix", "ERR", &e.to_string()]),
    }
    match ops::get_singleton(&det).unwrap() {
        None => out.line(&["singleton", "null"]),
        Some(s) => out.line(&["singleton", &esc_cps(&s)]),
    }
    let finite = ops::is_finite(&det).unwrap();
    if finite {
        if det.get_num_states() <= 60 {
            let topo: Vec<String> = ops::topo_sort_states(&det)
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect();
            out.line(&["topo", &topo.join(",")]);
        }
        let mut it = LimitedFiniteStringsIterator::new(&det, 50).unwrap();
        let mut strs = Vec::new();
        while let Some(s) = it.next_string().unwrap() {
            strs.push(esc_cps(&s));
        }
        let n = strs.len().to_string();
        let mut fields: Vec<&str> = vec!["strings", &n];
        fields.extend(strs.iter().map(String::as_str));
        out.line(&fields);
    }
    compiled(out, "compiled_det", &det, binary, probes, finite);
}

// ---- case families -------------------------------------------------------------

fn regexp(out: &mut Out, pattern: &str, syntax: i32, mflags: i32, probes: &[String]) {
    let re = match RegExp::with_flags(pattern, syntax, mflags) {
        Ok(r) => r,
        Err(e) => {
            out.line(&["parse", "ERR", &esc(&e.to_string())]);
            return;
        }
    };
    out.line(&["parse", "OK", &esc(&re.to_string())]);
    out.line(&["tree", &esc(&re.to_string_tree())]);
    let a = match re.to_automaton() {
        Ok(a) => a,
        Err(AutomatonError::TooComplex(e)) => {
            out.line(&["auto", "TOO_COMPLEX", &esc(&e.to_string())]);
            return;
        }
        Err(e) => {
            out.line(&["auto", "ERR", &esc(&e.to_string())]);
            return;
        }
    };
    block(out, &a, false, probes);
}

fn lev(out: &mut Out, args: &[String], probes: &[String]) {
    let word = &args[0];
    let n: i32 = args[1].parse().unwrap();
    let transpositions = args[2] == "1";
    let prefix = &args[3];
    match LevenshteinAutomata::new(word, transpositions).to_automaton_with_prefix(n, prefix) {
        None => out.line(&["lev", "null"]),
        Some(a) => block(out, &a, false, probes),
    }
}

fn re(s: &str) -> Automaton {
    RegExp::new(s).unwrap().to_automaton().unwrap()
}

fn op(out: &mut Out, args: &[String], probes: &[String]) {
    let name = args[0].as_str();
    let rest = &args[1..];
    let limit = DEFAULT_DETERMINIZE_WORK_LIMIT;
    let a: Result<Automaton, AutomatonError> = match name {
        "concat" | "union" => {
            let list: Vec<Automaton> = rest.iter().map(|s| re(s)).collect();
            let refs: Vec<&Automaton> = list.iter().collect();
            Ok(if name == "concat" {
                ops::concatenate(&refs)
            } else {
                ops::union(&refs)
            })
        }
        "intersection" => Ok(ops::intersection(&re(&rest[0]), &re(&rest[1]))),
        "minus" => ops::minus(&re(&rest[0]), &re(&rest[1]), limit).map_err(Into::into),
        "complement" => ops::complement(&re(&rest[0]), limit).map_err(Into::into),
        "optional" => Ok(ops::optional(&re(&rest[0]))),
        "repeat" => Ok(ops::repeat(&re(&rest[0]))),
        "repeatmin" => Ok(ops::repeat_min(&re(&rest[0]), rest[1].parse().unwrap())),
        "repeatrange" => Ok(ops::repeat_range(
            &re(&rest[0]),
            rest[1].parse().unwrap(),
            rest[2].parse().unwrap(),
        )),
        "reverse" => Ok(ops::reverse(&re(&rest[0]))),
        other => panic!("unknown op {other}"),
    };
    match a {
        Ok(a) => block(out, &a, false, probes),
        Err(e) => out.line(&["op", "TOO_COMPLEX", &esc(&e.to_string())]),
    }
}

fn bytes(h: &str) -> Option<Vec<u8>> {
    if h == "null" {
        return None;
    }
    Some(
        (0..h.len() / 2)
            .map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap())
            .collect(),
    )
}

fn automata_case(out: &mut Out, args: &[String], probes: &[String]) {
    let name = args[0].as_str();
    let rest = &args[1..];
    let int = |i: usize| rest[i].parse::<i32>().unwrap();
    let mut binary = false;
    let a: Result<Automaton, AutomatonError> = match name {
        "string" => Ok(automata::make_string(&rest[0])),
        "ci_string" => Ok(automata::make_case_insensitive_string(&rest[0])),
        "char_range" => Ok(automata::make_char_range(int(0), int(1))),
        "decimal" => automata::make_decimal_interval(int(0), int(1), int(2)),
        "any_string" => Ok(automata::make_any_string()),
        "any_char" => Ok(automata::make_any_char()),
        "empty" => Ok(automata::make_empty()),
        "empty_string" => Ok(automata::make_empty_string()),
        "char_set" => {
            let cps: Vec<i32> = rest.iter().map(|s| s.parse().unwrap()).collect();
            Ok(automata::make_char_set(&cps))
        }
        "any_binary" => {
            binary = true;
            Ok(automata::make_any_binary())
        }
        "non_empty_binary" => {
            binary = true;
            Ok(automata::make_non_empty_binary())
        }
        "binary" => {
            binary = true;
            Ok(automata::make_binary(&bytes(&rest[0]).unwrap()))
        }
        "binary_interval" => {
            binary = true;
            let (lo, hi) = (bytes(&rest[0]), bytes(&rest[2]));
            automata::make_binary_interval(
                lo.as_deref(),
                rest[1] == "1",
                hi.as_deref(),
                rest[3] == "1",
            )
        }
        "string_union" | "binary_string_union" => {
            binary = name.starts_with("binary");
            // Java joins zero inputs into one empty field.
            let none = rest.len() == 1 && rest[0].is_empty();
            let items: Vec<&[u8]> = if none {
                Vec::new()
            } else {
                rest.iter().map(|s| s.as_bytes()).collect()
            };
            if binary {
                automata::make_binary_string_union(items)
            } else {
                automata::make_string_union(items)
            }
        }
        other => panic!("unknown factory {other}"),
    };
    match a {
        Ok(a) => block(out, &a, binary, probes),
        Err(e) => out.line(&["automata", "ERR", &esc(&err_msg(&e))]),
    }
}

// ---- driver ----------------------------------------------------------------------

#[test]
fn automata_match_lucene() {
    let text = std::fs::read_to_string(fixture()).expect("fixtures/data/automata/automata.tsv");
    let mut lines = text.lines().peekable();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    assert_eq!(header[0], "PROBES");
    let global: Vec<String> = header[2..].iter().map(|s| unesc(s)).collect();
    assert_eq!(global.len(), header[1].parse::<usize>().unwrap());

    let mut cases = 0;
    let mut compared = 0;
    let mut failures: Vec<String> = Vec::new();
    let mut by_kind = std::collections::BTreeMap::<String, usize>::new();
    while let Some(case_line) = lines.next() {
        let case: Vec<&str> = case_line.split('\t').collect();
        assert_eq!(case[0], "CASE", "{case_line}");
        let probe_line: Vec<&str> = lines.next().unwrap().split('\t').collect();
        assert_eq!(probe_line[0], "probes");
        let extra: Vec<String> = probe_line[2..].iter().map(|s| unesc(s)).collect();
        let mut probes = global.clone();
        probes.extend(extra.iter().cloned());
        let mut expected = Vec::new();
        while let Some(l) = lines.peek() {
            if l.starts_with("CASE\t") {
                break;
            }
            expected.push(lines.next().unwrap().to_string());
        }
        let args: Vec<String> = case[2..].iter().map(|s| unesc(s)).collect();
        let mut out = Out(Vec::new());
        match case[1] {
            "regexp" => regexp(
                &mut out,
                &args[0],
                args[1].parse().unwrap(),
                args[2].parse().unwrap(),
                &global,
            ),
            "lev" => lev(&mut out, &args, &probes),
            "op" => op(&mut out, &args, &global),
            "automata" => automata_case(&mut out, &args, &global),
            other => panic!("unknown case kind {other}"),
        }
        cases += 1;
        *by_kind.entry(case[1].to_string()).or_default() += 1;
        compared += expected.len();
        if out.0 != expected {
            let n = out.0.len().max(expected.len());
            for i in 0..n {
                let (g, e) = (out.0.get(i), expected.get(i));
                if g != e {
                    failures.push(format!(
                        "{case_line}\n  line {i}\n  java: {}\n  rust: {}",
                        e.map_or("<none>", |s| s.as_str())
                            .chars()
                            .take(400)
                            .collect::<String>(),
                        g.map_or("<none>", |s| s.as_str())
                            .chars()
                            .take(400)
                            .collect::<String>()
                    ));
                    break;
                }
            }
        }
    }
    eprintln!("{cases} cases ({by_kind:?}), {compared} result lines compared");
    assert!(cases > 250, "fixture looks truncated: {cases} cases");
    assert!(
        failures.is_empty(),
        "{} of {cases} cases differ from Lucene:\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
