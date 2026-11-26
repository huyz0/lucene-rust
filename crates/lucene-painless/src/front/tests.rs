use super::ast::*;
use super::{parse, Limits};

fn script(src: &str) -> Script {
    parse(src, &Limits::default()).unwrap_or_else(|e| panic!("{src}: {e}"))
}

fn fails(src: &str) -> super::ParseError {
    parse(src, &Limits::default()).expect_err(src)
}

/// A fully parenthesised rendering of an expression.
fn show(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Num(_, s) | ExprKind::Var(s) => s.clone(),
        ExprKind::Bool(b) => b.to_string(),
        ExprKind::Null => "null".into(),
        ExprKind::Str(s) => format!("{s:?}"),
        ExprKind::Regex(p, f) => format!("/{p}/{f}"),
        ExprKind::List(v) => format!("[{}]", v.iter().map(show).collect::<Vec<_>>().join(",")),
        ExprKind::Map(v) => format!(
            "[{}]",
            v.iter()
                .map(|(k, v)| format!("{}:{}", show(k), show(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        ExprKind::CallLocal(n, a) => format!("{n}({})", args(a)),
        ExprKind::NewObject(t, a) => format!("new {}({})", t.name, args(a)),
        ExprKind::NewArray(t, d) => {
            format!(
                "new {}{}",
                t.name,
                d.iter()
                    .map(|d| format!("[{}]", show(d)))
                    .collect::<String>()
            )
        }
        ExprKind::NewArrayInit(t, v) => format!(
            "new {}{}{{{}}}",
            t.name,
            "[]".repeat(t.dims),
            v.iter().map(show).collect::<Vec<_>>().join(",")
        ),
        ExprKind::Call {
            recv,
            name,
            args: a,
            null_safe,
        } => {
            format!(
                "{}{}{name}({})",
                show(recv),
                if *null_safe { "?." } else { "." },
                args(a)
            )
        }
        ExprKind::Field {
            recv,
            name,
            null_safe,
        } => {
            format!(
                "{}{}{name}",
                show(recv),
                if *null_safe { "?." } else { "." }
            )
        }
        ExprKind::Index(a, i) => format!("{}[{}]", show(a), show(i)),
        ExprKind::Unary(op, e) => format!("({op:?} {})", show(e)),
        ExprKind::IncDec {
            prefix,
            inc,
            target,
        } => {
            format!(
                "({}{} {})",
                if *prefix { "pre" } else { "post" },
                if *inc { "++" } else { "--" },
                show(target)
            )
        }
        ExprKind::Cast(t, e) => format!("(cast {}{} {})", t.name, "[]".repeat(t.dims), show(e)),
        ExprKind::Binary(op, l, r) => format!("({op:?} {} {})", show(l), show(r)),
        ExprKind::InstanceOf(e, t) => format!("(instanceof {} {})", show(e), t.name),
        ExprKind::Cond(c, a, b) => format!("(? {} {} {})", show(c), show(a), show(b)),
        ExprKind::Assign(op, t, v) => format!("({op:?} {} {})", show(t), show(v)),
        ExprKind::Paren(e) => format!("<{}>", show(e)),
    }
}

fn args(a: &[Arg]) -> String {
    a.iter()
        .map(|a| match a {
            Arg::Expr(e) => show(e),
            Arg::Lambda(l) => format!(
                "\\{}->{}",
                l.params
                    .iter()
                    .map(|(_, n)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
                match &l.body {
                    LambdaBody::Expr(e) => show(e),
                    LambdaBody::Block(b) => format!("{{{}}}", b.stmts.len()),
                }
            ),
            Arg::FuncRef(r) => format!(
                "{}::{}",
                r.ty.as_ref().map_or("this", |t| t.name.as_str()),
                r.name.as_deref().unwrap_or("new")
            ),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn expr(src: &str) -> String {
    let s = script(src);
    match &s.body[0].kind {
        StmtKind::Expr(e) => show(e),
        StmtKind::Return(Some(e)) => show(e),
        k => panic!("{src}: {k:?}"),
    }
}

#[test]
fn binary_operators_climb_in_the_grammars_order() {
    assert_eq!(expr("a + b * c"), "(Add a (Mul b c))");
    assert_eq!(expr("a - b - c"), "(Sub (Sub a b) c)");
    assert_eq!(expr("a < b == c"), "(Eq (Lt a b) c)");
    assert_eq!(expr("a | b ^ c & d"), "(BitOr a (Xor b (BitAnd c d)))");
    assert_eq!(expr("a || b && c"), "(Or a (And b c))");
    assert_eq!(expr("a << b + c"), "(Lsh a (Add b c))");
    assert_eq!(expr("a =~ b + c"), "(Find a (Add b c))");
    assert_eq!(expr("a ?: b ?: c"), "(Elvis a (Elvis b c))");
    assert_eq!(expr("a || b ?: c"), "(Elvis (Or a b) c)");
}

#[test]
fn instanceof_sits_between_comparison_and_equality() {
    assert_eq!(
        expr("a instanceof String == true"),
        "(Eq (instanceof a String) true)"
    );
    assert_eq!(expr("a < b instanceof List"), "(instanceof (Lt a b) List)");
}

#[test]
fn conditional_and_assignment_are_right_associative() {
    assert_eq!(expr("a ? b : c ? d : e"), "(? a b (? c d e))");
    assert_eq!(expr("a = b = c"), "(Set a (Set b c))");
    assert_eq!(expr("a += b ? c : d"), "(With(Add) a (? b c d))");
    assert_eq!(expr("x >>>= 2"), "(With(Ush) x 2)");
}

#[test]
fn unary_cast_and_parenthesis() {
    assert_eq!(expr("-a * b"), "(Mul (Minus a) b)");
    assert_eq!(expr("(int) -a"), "(cast int (Minus a))");
    assert_eq!(expr("(int) a + b"), "(Add (cast int a) b)");
    assert_eq!(expr("(String) a"), "(cast String a)");
    assert_eq!(expr("(List) -a"), "(Sub <List> a)");
    assert_eq!(expr("(a) + b"), "(Add <a> b)");
    assert_eq!(expr("(int[]) a"), "(cast int[] a)");
    assert_eq!(expr("(def) a.b"), "(cast def a.b)");
    assert_eq!(expr("!a && ~b"), "(And (Not a) (BitNot b))");
    assert_eq!(expr("++a.b"), "(pre++ a.b)");
    assert_eq!(expr("a[0]--"), "(post-- a[0])");
}

#[test]
fn postfixes_initializers_and_creation() {
    assert_eq!(expr("a.b(1)?.c.0[2].d"), "a.b(1)?.c.0[2].d");
    assert_eq!(expr("[1, 2]"), "[1,2]");
    assert_eq!(expr("[]"), "[]");
    assert_eq!(expr("[:]"), "[]");
    assert_eq!(expr("['a': 1, 'b': 2]"), "[\"a\":1,\"b\":2]");
    assert_eq!(expr("new ArrayList()"), "new ArrayList()");
    assert_eq!(expr("new Map.Entry(a)"), "new Map.Entry(a)");
    assert_eq!(expr("new int[2][3]"), "new int[2][3]");
    assert_eq!(expr("new int[] {1, 2}"), "new int[]{1,2}");
    assert_eq!(expr("new int[][] {}"), "new int[][]{}");
    assert_eq!(expr("new int[2].length"), "new int[2].length");
    assert_eq!(expr("f(1).g"), "f(1).g");
}

#[test]
fn lambdas_and_method_references_are_arguments() {
    assert_eq!(expr("l.map(x -> x + 1)"), "l.map(\\x->(Add x 1))");
    assert_eq!(expr("l.map((x, y) -> x)"), "l.map(\\x,y->x)");
    assert_eq!(expr("l.map((int x) -> { return x; })"), "l.map(\\x->{1})");
    assert_eq!(expr("l.map(int x -> x)"), "l.map(\\x->x)");
    assert_eq!(expr("l.map(() -> 1)"), "l.map(\\->1)");
    assert_eq!(expr("l.map(Integer::parseInt)"), "l.map(Integer::parseInt)");
    assert_eq!(expr("l.map(ArrayList::new)"), "l.map(ArrayList::new)");
    assert_eq!(expr("l.map(this::f)"), "l.map(this::f)");
    assert_eq!(expr("f((x))"), "f(<x>)");
    assert!(parse("x -> x", &Limits::default()).is_err());
}

#[test]
fn strings_resolve_their_escapes_and_regexes_are_told_from_division() {
    assert_eq!(expr(r#"'it\'s \\ "q"'"#), "\"it's \\\\ \\\"q\\\"\"");
    assert_eq!(expr("a / b / c"), "(Div (Div a b) c)");
    assert_eq!(expr("x /= 2"), "(With(Div) x 2)");
    assert_eq!(expr("a =~ /b+/i"), "(Find a /b+/i)");
    assert_eq!(expr("(a) / 2"), "(Div <a> 2)");
    assert_eq!(expr("f(/a\\/b/)"), "f(/a\\/b/)");
    assert!(fails(r"'a\nb'").message.contains("escape"));
}

#[test]
fn numbers_take_the_longest_match() {
    let kinds = |src: &str| match &script(src).body[0].kind {
        StmtKind::Expr(Expr {
            kind: ExprKind::Num(k, s),
            ..
        }) => (*k, s.clone()),
        k => panic!("{k:?}"),
    };
    assert_eq!(kinds("017"), (NumKind::Octal, "017".into()));
    assert_eq!(kinds("0xFFL"), (NumKind::Hex, "0xFFL".into()));
    assert_eq!(kinds("12L"), (NumKind::Integer, "12L".into()));
    assert_eq!(kinds("1f"), (NumKind::Integer, "1f".into()));
    assert_eq!(kinds("1.5e-3d"), (NumKind::Decimal, "1.5e-3d".into()));
    assert_eq!(kinds("1e5"), (NumKind::Decimal, "1e5".into()));
    // `1.foo` is an integer, a dot and a name: no fraction without digits.
    assert_eq!(expr("1.foo()"), "1.foo()");
}

#[test]
fn statements() {
    let s = script(
        "int add(int a, int b) { return a + b } \
         int x = 1, y; \
         if (x > 0) x++; else { x-- } \
         for (int i = 0; i < 3; i++) x += i; \
         for (def v : [1]) { continue } \
         for (v in [1]) break; \
         while (x < 10) x++; \
         while (false); \
         do { x++ } while (x < 5); \
         try { throw new RuntimeException('e') } catch (RuntimeException e) { x = 0 } catch (Exception e) { } \
         return x",
    );
    assert_eq!(s.functions.len(), 1);
    assert_eq!(s.functions[0].name, "add");
    assert_eq!(s.body.len(), 10);
    assert!(matches!(&s.body[0].kind, StmtKind::Decl(d) if d.len() == 2));
    assert!(matches!(
        &s.body[1].kind,
        StmtKind::If {
            otherwise: Some(_),
            ..
        }
    ));
    assert!(matches!(
        &s.body[3].kind,
        StmtKind::Each { ty: Some(_), .. }
    ));
    assert!(matches!(&s.body[4].kind, StmtKind::Each { ty: None, .. }));
    assert!(matches!(
        &s.body[6].kind,
        StmtKind::While { body: None, .. }
    ));
    assert!(matches!(&s.body[8].kind, StmtKind::Try { catches, .. } if catches.len() == 2));
}

#[test]
fn a_dangling_else_binds_to_the_nearest_if() {
    let s = script("if (a) if (b) x(); else y();");
    let StmtKind::If {
        then, otherwise, ..
    } = &s.body[0].kind
    else {
        panic!()
    };
    assert!(otherwise.is_none());
    assert!(matches!(
        &then.kind,
        StmtKind::If {
            otherwise: Some(_),
            ..
        }
    ));
}

#[test]
fn a_declaration_is_told_from_an_expression_by_its_name() {
    assert!(matches!(
        &script("Map.Entry e = null").body[0].kind,
        StmtKind::Decl(_)
    ));
    assert!(matches!(&script("int[][] a").body[0].kind, StmtKind::Decl(d) if d[0].ty.dims == 2));
    assert!(matches!(
        &script("a[0] = 1").body[0].kind,
        StmtKind::Expr(_)
    ));
    assert!(matches!(
        &script("Integer.MAX_VALUE").body[0].kind,
        StmtKind::Expr(_)
    ));
}

#[test]
fn refusals_say_where_and_what() {
    assert_eq!(fails("int x = ;").offset, 8);
    assert!(fails("x +").message.contains("expression"));
    assert!(fails("if (x) y 1").message.contains(';'));
    assert!(fails("a.\nb").message.contains("name"));
    assert!(fails("a ? b").message.contains(':'));
    assert!(fails("'abc").message.contains("unterminated"));
    assert!(fails("/* x").message.contains("unterminated"));
    assert!(fails("try { }").message.contains("catch"));
    assert!(fails("x = 1 #").message.contains("unexpected character"));
    assert_eq!(fails("{ x }").offset, 0);
}

#[test]
fn comments_are_skipped() {
    assert_eq!(expr("1 /* c */ + // d\n 2"), "(Add 1 2)");
}

#[test]
fn bounds_refuse_before_anything_runs() {
    let deep = format!("{}1{}", "(".repeat(150), ")".repeat(150));
    assert!(fails(&deep).message.contains("nests deeper"));
    let ok = format!("{}1{}", "(".repeat(20), ")".repeat(20));
    script(&ok);
    let big = " ".repeat(65_536);
    assert!(fails(&big).message.contains("over the limit"));
    let lim = Limits {
        max_source: 4,
        max_depth: 100,
    };
    assert!(parse("1 + 2 + 3", &lim).is_err());
    let blocks = format!("{}x(){}", "if (a) {".repeat(150), "}".repeat(150));
    assert!(fails(&blocks).message.contains("nests deeper"));
}
