//! The allocation limits of one invocation (`lucene_painless::arena`): a
//! script that builds past them fails with a `PainlessError` and the next
//! run on the thread starts clean.

use std::rc::Rc;

use lucene_painless::arena;
use lucene_painless::compile::compile;
use lucene_painless::front::{analyze, parse, Limits};
use lucene_painless::jmap::{JMap, Order};
use lucene_painless::value::{MapKind, Value};
use lucene_painless::vm::{ParamsEnv, Vm};

fn run(src: &str) -> Result<Value, String> {
    let ast = parse(src, &Limits::default()).expect("parses");
    let script = analyze(&ast).expect("lowers");
    let prog = Rc::new(compile(&script).expect("compiles"));
    let mut env = ParamsEnv {
        params: Value::map(MapKind::HashMap, JMap::new(Order::Hash)),
    };
    Vm::new(prog, &mut env)
        .execute(&[])
        .map_err(|t| t.message().unwrap_or_default())
}

/// One test: the limits are process-wide.
#[test]
fn a_script_past_its_allocation_limits_fails_and_the_next_run_is_clean() {
    arena::set_limits(arena::DEFAULT_INVOCATION, arena::DEFAULT_VALUE);
    // A string over 1 MiB: doubled to 2 MiB.
    let doubled =
        "String s = 'xxxxxxxx'; for (int i = 0; i < 18; i++) { s += s; } return s.length();";
    let e = run(doubled).expect_err("past the value limit");
    assert!(e.contains("allocated more than"), "{e}");
    // A script that stays inside is untouched, and runs after the failure.
    let small =
        "String s = 'xxxxxxxx'; for (int i = 0; i < 10; i++) { s += s; } return s.length();";
    let v = run(small).expect("inside the limits");
    assert!(format!("{v:?}").contains("8192"), "{v:?}");
    // Many strings under 1 MiB add up past 4 MiB.
    let many = "int n = 0; for (int i = 0; i < 100; i++) { String s = 'x'; for (int j = 0; j < 16; j++) { s += s; } n += s.length(); } return n;";
    assert!(run(many)
        .expect_err("past the invocation limit")
        .contains("allocated more than"));
    // The limits are settable.
    arena::set_limits(1 << 30, 1 << 30);
    assert!(run(doubled).is_ok());
    arena::set_limits(arena::DEFAULT_INVOCATION, arena::DEFAULT_VALUE);
}

/// A script over 20,000 IR nodes is refused when it is compiled, with the
/// count, and one just under it compiles.
#[test]
fn a_script_over_the_ir_node_cap_is_refused_at_compile_time() {
    use lucene_painless::front::source::{compile, Refused, MAX_IR_NODES};
    assert_eq!(MAX_IR_NODES, 20_000);
    let body = |n: usize| format!("int a = 0; {} return a;", "a += 1;".repeat(n));
    let small = compile(&body(10), 0);
    assert!(small.is_ok(), "{small:?}");
    match compile(&body(9_000), 0) {
        Err(Refused::Compile { message, .. }) => {
            assert!(message.contains("over the limit of 20000"), "{message}");
        }
        other => panic!("{other:?}"),
    }
}
