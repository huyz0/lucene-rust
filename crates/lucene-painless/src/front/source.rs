//! Script source to the IR a node runs: what an engine does with the source
//! of a `script` it is given, and which document fields that script reads.

use crate::front::{
    analyze_returning, parse, Binding, Limits, Refusal, AGG, DOC, INGEST, METRIC_MAP,
    METRIC_REDUCE, METRIC_STATE, SCORE,
};
use crate::ir::{Member, Node, Script};

/// Why source is not a script this runtime runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// Painless would reject it: the message and where.
    Compile { offset: usize, message: String },
    /// A construct the front end does not lower yet.
    Unsupported(String),
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refused::Compile { offset, message } => write!(f, "{message} at offset {offset}"),
            Refused::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

/// The variables a script of engine context `context` reads (`0` and `1`
/// score, `2` filter, `3` and `4` number sort, `5` aggregation, `6`
/// `terms_set`, `10` an ingest script processor, `11` to `14` a `scripted_metric`'s map, init, combine and reduce); `None` for a context the front end does not serve.
pub fn bindings(context: u8) -> Option<&'static [Binding]> {
    Some(match context {
        0 | 1 | 4 => SCORE,
        2 | 3 | 6 => DOC,
        5 => AGG,
        10 => INGEST,
        11 => METRIC_MAP,
        12 | 13 => METRIC_STATE,
        14 => METRIC_REDUCE,
        _ => return None,
    })
}

/// Lowers `source` as a script of engine context `context`.
///
/// # Errors
/// [`Refused`] when it does not parse, Painless would reject it, or it uses
/// something not served yet.
pub fn compile(source: &str, context: u8) -> Result<Script, Refused> {
    let bindings = bindings(context)
        .ok_or_else(|| Refused::Unsupported(format!("script context {context}")))?;
    let ast = parse(source, &Limits::default()).map_err(|e| Refused::Compile {
        offset: e.offset,
        message: e.message,
    })?;
    // The type `execute` returns: a score or a sort key is a `double`, a
    // filter a `boolean`.
    let ret = match context {
        0 | 1 | 3 | 4 => "double",
        2 => "boolean",
        10..=12 => "void",
        _ => crate::types::DEF,
    };
    let script = analyze_returning(&ast, bindings, ret).map_err(|r| match r {
        Refusal::Compile { offset, message } => Refused::Compile {
            offset: offset as usize,
            message,
        },
        Refusal::Unsupported(what) => Refused::Unsupported(what),
    })?;
    let mut nodes = 0usize;
    for f in &script.functions {
        any_node(&f.body, &mut |_| nodes += 1);
    }
    if nodes > MAX_IR_NODES {
        return Err(Refused::Compile {
            offset: 0,
            message: format!("the script has {nodes} operations, over the limit of {MAX_IR_NODES}"),
        });
    }
    Ok(script)
}

/// The most IR nodes a script may lower to.
pub const MAX_IR_NODES: usize = 20_000;

/// The fields the script reads through a constant name (`doc['f']`,
/// `doc.get('f')`, `doc.containsKey('f')`), in the order it first reads them.
pub fn doc_fields(script: &Script) -> Vec<String> {
    let mut out = Vec::new();
    for f in &script.functions {
        fields(&f.body, &mut out);
    }
    out
}

/// Whether the script reads `_score`.
pub fn reads_score(script: &Script) -> bool {
    let mut found = false;
    for f in &script.functions {
        any_node(&f.body, &mut |n| {
            if let Node::InvokeCallMember {
                member: Member::Local { name, internal, .. },
                ..
            } = n
            {
                found |= *internal && name == "get_score";
            }
        });
    }
    found
}

fn is_doc(n: &Node) -> bool {
    match n {
        Node::At(_, _, c) => is_doc(c),
        Node::InvokeCallMember {
            member: Member::Local { name, internal, .. },
            ..
        } => *internal && name == "getDoc",
        _ => false,
    }
}

fn key(n: &Node) -> Option<&str> {
    match n {
        Node::At(_, _, c) => key(c),
        Node::Constant {
            value: crate::ir::Constant::Str(s),
            ..
        } => Some(s),
        _ => None,
    }
}

fn fields(n: &Node, out: &mut Vec<String>) {
    any_node(n, &mut |n| {
        let Node::BinaryImpl { left, right, .. } = n else {
            return;
        };
        let name = match (&**left, &**right) {
            // `doc['f']`: the document, then the key, then the map shortcut.
            (
                Node::BinaryImpl {
                    left: d, right: k, ..
                },
                Node::LoadMapShortcut { .. },
            ) if is_doc(d) => key(k),
            // `doc.get('f')` and `doc.containsKey('f')`.
            (d, Node::InvokeCall { method, args, .. })
                if is_doc(d)
                    && method
                        .as_ref()
                        .is_some_and(|m| matches!(m.name.as_str(), "get" | "containsKey")) =>
            {
                args.first().and_then(key)
            }
            _ => None,
        };
        if let Some(name) = name {
            if !out.iter().any(|o| o == name) {
                out.push(name.to_string());
            }
        }
    });
}

/// Calls `f` on `n` and every node under it, parents first.
fn any_node(n: &Node, f: &mut dyn FnMut(&Node)) {
    use Node::*;
    f(n);
    let mut each = |c: &Node| any_node(c, f);
    match n {
        At(_, _, c) | StatementExpression(c) | Throw(c) => each(c),
        Block(v) | DeclarationBlock(v) | StringConcat { args: v, .. } => v.iter().for_each(each),
        Declaration { init: c, .. } | While { cond: c, .. } | DoWhile { cond: c, .. } => {
            if let Some(c) = c {
                each(c)
            }
            if let While { body: Some(b), .. } | DoWhile { body: Some(b), .. } = n {
                each(b)
            }
        }
        Return(c) => {
            if let Some(c) = c {
                each(c)
            }
        }
        If { cond, then } => {
            each(cond);
            each(then);
        }
        IfElse {
            cond,
            then,
            otherwise,
        } => {
            each(cond);
            each(then);
            each(otherwise);
        }
        For {
            init,
            cond,
            after,
            body,
            ..
        } => {
            for c in [init, cond, after, body].into_iter().flatten() {
                each(c)
            }
        }
        ForEachArray { iterable, body, .. } | ForEachIterable { iterable, body, .. } => {
            each(iterable);
            if let Some(b) = body {
                each(b)
            }
        }
        Try { body, catches } => {
            each(body);
            catches.iter().for_each(each);
        }
        Catch { body, .. } => {
            if let Some(b) = body {
                each(b)
            }
        }
        BinaryImpl { left, right, .. }
        | BinaryMath { left, right, .. }
        | Boolean { left, right, .. }
        | Comparison { left, right, .. }
        | Elvis { left, right, .. } => {
            each(left);
            each(right);
        }
        Conditional {
            cond, left, right, ..
        } => {
            each(cond);
            each(left);
            each(right);
        }
        Cast { child, .. }
        | Instanceof { child, .. }
        | StoreVariable { child, .. }
        | UnaryMath { child, .. }
        | Dup { child, .. }
        | FlipArrayIndex { child, .. }
        | FlipCollectionIndex { child, .. }
        | FlipDefIndex { child, .. }
        | StoreBrace { child, .. }
        | StoreBraceDef { child, .. }
        | StoreFieldMember { child, .. }
        | StoreDotDef { child, .. }
        | StoreDot { child, .. }
        | StoreDotShortcut { child, .. }
        | StoreListShortcut { child, .. }
        | StoreMapShortcut { child, .. }
        | NullSafeSub { child, .. } => each(child),
        InvokeCall { args, .. }
        | InvokeCallDef { args, .. }
        | InvokeCallMember { args, .. }
        | ListInit { args, .. }
        | NewArray { args, .. }
        | NewObject { args, .. } => args.iter().for_each(each),
        MapInit { keys, values, .. } => {
            keys.iter().for_each(&mut each);
            values.iter().for_each(each);
        }
        Break
        | Continue
        | Constant { .. }
        | Null { .. }
        | LoadVariable { .. }
        | LoadBrace { .. }
        | LoadBraceDef { .. }
        | LoadDotArrayLength { .. }
        | LoadDotDef { .. }
        | LoadDot { .. }
        | LoadDotShortcut { .. }
        | LoadListShortcut { .. }
        | LoadMapShortcut { .. }
        | Static { .. }
        | LoadFieldMember { .. }
        | TypedInterfaceReference { .. }
        | DefInterfaceReference { .. }
        | TypedCaptureReference { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower(src: &str) -> Script {
        compile(src, 1).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    #[test]
    fn the_fields_a_script_reads_by_constant_name() {
        let s = lower(
            "double a = doc['price'].value; if (doc.containsKey('qty')) { a += doc.get('qty').value; } a + doc['price'].value + params.x",
        );
        assert_eq!(doc_fields(&s), ["price", "qty"]);
        assert!(doc_fields(&lower("params.x + 1")).is_empty());
    }

    #[test]
    fn a_script_reads_the_score_or_does_not() {
        assert!(reads_score(&lower("_score * 2")));
        assert!(!reads_score(&lower("doc['f'].value")));
    }

    #[test]
    fn what_a_script_returns_is_cast_to_its_contexts_type() {
        // A filter returns a boolean, a score or a sort key a double.
        assert!(compile("doc['f'].value > 1", 2).is_ok());
        assert!(compile("1 + 1", 2).is_err());
        assert!(compile("return 'a';", 2).is_err());
        assert!(compile("doc['f'].value", 0).is_ok());
        assert!(compile("return 'a';", 0).is_err());
        assert!(compile("return true;", 3).is_err());
        // `def` casts at run time; an aggregation's script returns any object.
        assert!(compile("params.x", 2).is_ok());
        let r = compile("return 'a';", 5);
        assert!(r.is_ok(), "{r:?}");
    }

    #[test]
    fn a_context_without_bindings_or_a_bad_script_is_refused() {
        assert!(matches!(compile("1", 7), Err(Refused::Unsupported(_))));
        assert!(matches!(compile("1 +", 1), Err(Refused::Compile { .. })));
        assert!(matches!(
            compile("int x = 'a';", 1),
            Err(Refused::Compile { .. })
        ));
        // `_score` is not a variable of a filter.
        assert!(compile("_score", 2).is_err());
    }
}
