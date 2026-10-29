//! Lucene 10.6 (`branch_10_6`, #16423): `JavascriptCompiler`'s nesting
//! guard. Every row of [`JAVA`] is what Java's compiler of that branch
//! (compiled against the 10.5.2 jars) answers; `TestJavascriptCompiler`'s
//! `testRecursionDepth*` follow, with where this port answers otherwise.

use lucene_search::expressions::{
    compile, compile_with_max_nesting_depth, default_functions, Error, ParseException,
    DEFAULT_MAX_NESTING_DEPTH,
};

/// `(maxNestingDepth, source, outcome)`: Java's, each row printed by
/// `JavascriptCompiler.compile(source, DEFAULT_FUNCTIONS, maxNestingDepth)`
/// of `branch_10_6` (#16423) run over the 10.5.2 jars: `None` compiles,
/// `Some((offset, limit))` is the nesting `ParseException` (`limit` is what
/// its message prints: `maxNestingDepth + 1` from the parser's listener,
/// `maxNestingDepth` from the compiler's), `Some((offset, -1))` the arity one.
type JavaRow = (i32, &'static str, Option<(usize, i32)>);

const JAVA: &[JavaRow] = &[
    (-1, "(((1)))", Some((0, 0))),
    (-1, "---1", Some((0, 0))),
    (-1, "1+2+3+4+5", Some((0, 0))),
    (-1, "1?2:3?4:5", Some((0, 0))),
    (-1, "abs(abs(1))", Some((0, 0))),
    (-1, "1*2+3", Some((0, 0))),
    (-1, "(1+2)*3+4", Some((0, 0))),
    (-1, " 1 + (2)", Some((1, 0))),
    (-1, "-(1+2)", Some((0, 0))),
    (-1, "1+(2+(3+(4)))", Some((0, 0))),
    (-1, "1<2&&3>4||!5", Some((0, 0))),
    (-1, "max(1, (2), -3)", Some((0, 0))),
    (-1, "1?(2?3:4):5", Some((0, 0))),
    (-1, "a.b[2] + c", Some((0, 0))),
    (-1, "1-(1-(1-(1-1)))", Some((0, 0))),
    (-1, "~~(~1)", Some((0, 0))),
    (
        -1,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((0, 0)),
    ),
    (0, "(((1)))", Some((0, 1))),
    (0, "---1", Some((0, 1))),
    (0, "1+2+3+4+5", Some((0, 1))),
    (0, "1?2:3?4:5", Some((0, 1))),
    (0, "abs(abs(1))", Some((0, 1))),
    (0, "1*2+3", Some((0, 1))),
    (0, "(1+2)*3+4", Some((0, 1))),
    (0, " 1 + (2)", Some((1, 1))),
    (0, "-(1+2)", Some((0, 1))),
    (0, "1+(2+(3+(4)))", Some((0, 1))),
    (0, "1<2&&3>4||!5", Some((0, 1))),
    (0, "max(1, (2), -3)", Some((0, 1))),
    (0, "1?(2?3:4):5", Some((0, 1))),
    (0, "a.b[2] + c", Some((0, 1))),
    (0, "1-(1-(1-(1-1)))", Some((0, 1))),
    (0, "~~(~1)", Some((0, 1))),
    (
        0,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((0, 1)),
    ),
    (1, "(((1)))", Some((1, 2))),
    (1, "---1", Some((1, 2))),
    (1, "1+2+3+4+5", Some((2, 2))),
    (1, "1?2:3?4:5", Some((2, 2))),
    (1, "abs(abs(1))", Some((4, 2))),
    (1, "1*2+3", Some((2, 2))),
    (1, "(1+2)*3+4", Some((1, 2))),
    (1, " 1 + (2)", Some((5, 2))),
    (1, "-(1+2)", Some((1, 2))),
    (1, "1+(2+(3+(4)))", Some((2, 2))),
    (1, "1<2&&3>4||!5", Some((2, 2))),
    (1, "max(1, (2), -3)", Some((4, 2))),
    (1, "1?(2?3:4):5", Some((2, 2))),
    (1, "a.b[2] + c", Some((9, 2))),
    (1, "1-(1-(1-(1-1)))", Some((2, 2))),
    (1, "~~(~1)", Some((1, 2))),
    (
        1,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((2, 2)),
    ),
    (2, "(((1)))", Some((2, 3))),
    (2, "---1", Some((2, 3))),
    (2, "1+2+3+4+5", Some((0, 2))),
    (2, "1?2:3?4:5", Some((6, 3))),
    (2, "abs(abs(1))", Some((8, 3))),
    (2, "1*2+3", Some((0, 2))),
    (2, "(1+2)*3+4", Some((3, 3))),
    (2, " 1 + (2)", Some((6, 3))),
    (2, "-(1+2)", Some((2, 3))),
    (2, "1+(2+(3+(4)))", Some((3, 3))),
    (2, "1<2&&3>4||!5", Some((7, 3))),
    (2, "max(1, (2), -3)", Some((8, 3))),
    (2, "1?(2?3:4):5", Some((3, 3))),
    (2, "a.b[2] + c", None),
    (2, "1-(1-(1-(1-1)))", Some((3, 3))),
    (2, "~~(~1)", Some((2, 3))),
    (
        2,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((4, 3)),
    ),
    (3, "(((1)))", Some((3, 4))),
    (3, "---1", Some((3, 4))),
    (3, "1+2+3+4+5", Some((0, 3))),
    (3, "1?2:3?4:5", None),
    (3, "abs(abs(1))", None),
    (3, "1*2+3", None),
    (3, "(1+2)*3+4", Some((1, 3))),
    (3, " 1 + (2)", None),
    (3, "-(1+2)", Some((4, 4))),
    (3, "1+(2+(3+(4)))", Some((5, 4))),
    (3, "1<2&&3>4||!5", Some((0, 3))),
    (3, "max(1, (2), -3)", Some((0, -1))),
    (3, "1?(2?3:4):5", Some((5, 4))),
    (3, "a.b[2] + c", None),
    (3, "1-(1-(1-(1-1)))", Some((5, 4))),
    (3, "~~(~1)", Some((3, 4))),
    (
        3,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((0, 3)),
    ),
    (4, "(((1)))", None),
    (4, "---1", None),
    (4, "1+2+3+4+5", Some((0, 4))),
    (4, "1?2:3?4:5", None),
    (4, "abs(abs(1))", None),
    (4, "1*2+3", None),
    (4, "(1+2)*3+4", Some((1, 4))),
    (4, " 1 + (2)", None),
    (4, "-(1+2)", None),
    (4, "1+(2+(3+(4)))", Some((6, 5))),
    (4, "1<2&&3>4||!5", None),
    (4, "max(1, (2), -3)", Some((0, -1))),
    (4, "1?(2?3:4):5", None),
    (4, "a.b[2] + c", None),
    (4, "1-(1-(1-(1-1)))", Some((6, 5))),
    (4, "~~(~1)", Some((4, 5))),
    (
        4,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((0, 4)),
    ),
    (5, "(((1)))", None),
    (5, "---1", None),
    (5, "1+2+3+4+5", None),
    (5, "1?2:3?4:5", None),
    (5, "abs(abs(1))", None),
    (5, "1*2+3", None),
    (5, "(1+2)*3+4", None),
    (5, " 1 + (2)", None),
    (5, "-(1+2)", None),
    (5, "1+(2+(3+(4)))", Some((8, 6))),
    (5, "1<2&&3>4||!5", None),
    (5, "max(1, (2), -3)", Some((0, -1))),
    (5, "1?(2?3:4):5", None),
    (5, "a.b[2] + c", None),
    (5, "1-(1-(1-(1-1)))", Some((8, 6))),
    (5, "~~(~1)", None),
    (
        5,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((0, 5)),
    ),
    (6, "(((1)))", None),
    (6, "---1", None),
    (6, "1+2+3+4+5", None),
    (6, "1?2:3?4:5", None),
    (6, "abs(abs(1))", None),
    (6, "1*2+3", None),
    (6, "(1+2)*3+4", None),
    (6, " 1 + (2)", None),
    (6, "-(1+2)", None),
    (6, "1+(2+(3+(4)))", Some((9, 7))),
    (6, "1<2&&3>4||!5", None),
    (6, "max(1, (2), -3)", Some((0, -1))),
    (6, "1?(2?3:4):5", None),
    (6, "a.b[2] + c", None),
    (6, "1-(1-(1-(1-1)))", Some((9, 7))),
    (6, "~~(~1)", None),
    (
        6,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((0, 6)),
    ),
    (7, "(((1)))", None),
    (7, "---1", None),
    (7, "1+2+3+4+5", None),
    (7, "1?2:3?4:5", None),
    (7, "abs(abs(1))", None),
    (7, "1*2+3", None),
    (7, "(1+2)*3+4", None),
    (7, " 1 + (2)", None),
    (7, "-(1+2)", None),
    (7, "1+(2+(3+(4)))", None),
    (7, "1<2&&3>4||!5", None),
    (7, "max(1, (2), -3)", Some((0, -1))),
    (7, "1?(2?3:4):5", None),
    (7, "a.b[2] + c", None),
    (7, "1-(1-(1-(1-1)))", Some((11, 8))),
    (7, "~~(~1)", None),
    (
        7,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((0, 7)),
    ),
    (8, "(((1)))", None),
    (8, "---1", None),
    (8, "1+2+3+4+5", None),
    (8, "1?2:3?4:5", None),
    (8, "abs(abs(1))", None),
    (8, "1*2+3", None),
    (8, "(1+2)*3+4", None),
    (8, " 1 + (2)", None),
    (8, "-(1+2)", None),
    (8, "1+(2+(3+(4)))", None),
    (8, "1<2&&3>4||!5", None),
    (8, "max(1, (2), -3)", Some((0, -1))),
    (8, "1?(2?3:4):5", None),
    (8, "a.b[2] + c", None),
    (8, "1-(1-(1-(1-1)))", None),
    (8, "~~(~1)", None),
    (
        8,
        "1+2*3-4/5%6<<7>8==9&10^11|12&&13||14?15:16",
        Some((0, 8)),
    ),
];

fn outcome(src: &str, max: i32) -> Option<ParseException> {
    match compile_with_max_nesting_depth(src, &default_functions(), max) {
        Ok(_) => None,
        Err(Error::Parse(p)) => Some(p),
        Err(e) => panic!("{src} at {max}: {e:?}"),
    }
}

#[test]
fn nesting_errors_are_javas() {
    for &(max, src, want) in JAVA {
        let got = outcome(src, max);
        match want {
            None => assert!(got.is_none(), "{src} at {max}: {got:?}"),
            Some((offset, -1)) => {
                let p = got.unwrap_or_else(|| panic!("{src} at {max} compiled"));
                assert_eq!(p.error_offset, offset, "{src} at {max}: {p:?}");
                assert!(p.message.contains("Expected (2) arguments"), "{p:?}");
            }
            Some((offset, limit)) => {
                let p = got.unwrap_or_else(|| panic!("{src} at {max} compiled"));
                assert_eq!(
                    p.message,
                    format!("Invalid expression '{src}': Nesting level too deep (>{limit})"),
                    "at {max}"
                );
                assert_eq!(p.error_offset, offset, "{src} at {max}");
            }
        }
    }
    // `Integer.MAX_VALUE + 1` wraps: nothing compiles, as in Java.
    let p = outcome("1", i32::MAX).unwrap();
    assert_eq!(
        (p.error_offset, p.message.as_str()),
        (
            0,
            "Invalid expression '1': Nesting level too deep (>-2147483648)"
        )
    );
}

fn sum(terms: usize) -> String {
    (0..terms)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join("+")
}

/// `testRecursionDepth1`, `1a` and `2` at the default limit: Java's listener
/// refuses the 1,025th nested expression (offset 1,024); this port's stack
/// bound (`MAX_DEPTH`) refuses far shallower trees first, with the message
/// Java 10.6 gives a stack overflow. Java accepts 1,023 parentheses; this
/// port refuses them, as it did under 10.5.
#[test]
fn recursion_depth_at_the_default_limit() {
    assert_eq!(DEFAULT_MAX_NESTING_DEPTH, 1024);
    for src in [
        format!("{}1{}", "(".repeat(20_000), ")".repeat(20_000)),
        format!("{}1{}", "(".repeat(1024), ")".repeat(1024)),
        format!("{}1{}", "(".repeat(1023), ")".repeat(1023)),
        format!("{}1", "-".repeat(20_000)),
    ] {
        match compile(&src) {
            Err(Error::Parse(p)) => assert_eq!(
                p.message,
                format!("Invalid expression '{src}': Nesting level too deep")
            ),
            other => panic!("{:?}", other.map(|e| e.source_text.len())),
        }
    }
}

/// `testRecursionDepth3`/`3a`: Java's compiler counts each `+` of a flat
/// sum as a level (1,025 terms fail at offset 0); this port flattens a run
/// of more than 16 operators of one level to one, so a flat sum of any
/// length compiles, as Lucene 10.5 compiles one of up to about 2,300 terms.
/// A run of 16 or fewer counts as Java's does.
#[test]
fn a_long_flat_sum_is_one_level() {
    assert!(compile(&sum(DEFAULT_MAX_NESTING_DEPTH as usize + 1)).is_ok());
    assert!(compile(&sum(20_000)).is_ok());
    // Java: offset 0, "(>10)" and "(>20)" for 30 terms.
    assert!(outcome(&sum(30), 20).is_none());
    assert!(outcome(&sum(30), 10).is_none());
    // Seventeen terms (sixteen operators) nest as Java's tree does.
    let p = outcome(&sum(17), 16).unwrap();
    assert_eq!((p.error_offset, p.message.ends_with("(>16)")), (0, true));
    assert!(outcome(&sum(17), 17).is_none());
}
