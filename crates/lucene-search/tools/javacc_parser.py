#!/usr/bin/env python3
"""Translate the mechanical half of a JavaCC 7 generated parser into Rust on
`crates/lucene-search/src/queryparser/javacc.rs`'s runtime: the token kinds
and `tokenImage` (`*Constants.java`), the `jj_la1` masks, and every syntactic
lookahead (`jj_3_<n>`, `jj_3R_*`), which only scan tokens. The productions,
which build the query tree, are ported by hand.

Usage:
    javacc_parser.py CONSTANTS.java PARSER.java constants > constants.rs
    javacc_parser.py CONSTANTS.java PARSER.java lookahead > lookahead.rs

The output still wants `cargo fmt`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from javacc_lexer import Parser, find_methods, outer, strip_comments, tokenize  # noqa: E402


def constants(src):
    """`int NAME = n;` in order, and `tokenImage`."""
    kinds = re.findall(r"\bint\s+(\w+)\s*=\s*(\d+);", src)
    m = re.search(r"tokenImage\s*=\s*\{(.*?)\};", src, re.S)
    images = re.findall(r'"((?:\\.|[^"\\])*)"', m.group(1))
    return [(n, int(v)) for n, v in kinds], images


def snake(name):
    """`jj_3R_IntervalText_685_5_33` -> `jj_3r_interval_text_685_5_33`."""
    head, _, rest = name.partition("jj_3R_")
    if not rest:
        return name
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", rest).lower()
    return "jj_3r_" + s


def expr(toks):
    out = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok == "jj_scan_token":
            k = toks.index(")", i)
            arg = "".join(toks[i + 2 : k])
            out.append(f"rt.scan_token({arg})?")
            i = k + 1
            continue
        if re.match(r"^jj_3", tok) and i + 2 < len(toks) and toks[i + 1] == "(" and toks[i + 2] == ")":
            out.append(f"{snake(tok)}(rt)?")
            i += 3
            continue
        if tok == "jj_scanpos":
            out.append("rt.jj_scanpos")
        else:
            out.append(tok)
        i += 1
    return " ".join(out).replace("! ", "!")


def stmts(ss, ind, last=False):
    p = "    " * ind
    out = []
    for k, s in enumerate(ss):
        kind = s[0]
        tail = last and k == len(ss) - 1
        if kind == "decl":
            if s[3] is not None:
                out.append(p + f"let {s[2]} = {expr(outer(s[3]))};")
            continue
        if kind == "expr":
            toks = s[1]
            if toks == ["Token", "xsp"]:
                continue  # each save is a `let`
            if toks[:2] == ["xsp", "="]:
                out.append(p + f"let xsp = {expr(toks[2:])};")
            else:
                out.append(p + expr(toks) + ";")
            continue
        if kind == "return":
            v = "Ok(" + expr(outer(s[1])) + ")"
            out.append(p + (v if tail else f"return {v};"))
            continue
        if kind == "break":
            out.append(p + "break;")
            continue
        if kind == "if":
            _, cond, then, els = s
            out.append(p + f"if {expr(outer(cond))} {{")
            out += stmts(then[1] if then[0] == "block" else [then], ind + 1)
            if els is None:
                out.append(p + "}")
            else:
                out.append(p + "} else {")
                out += stmts(els[1] if els[0] == "block" else [els], ind + 1)
                out.append(p + "}")
            continue
        if kind == "loop" or (kind == "while"):
            body = s[1]
            out.append(p + "loop {")
            out += stmts(body[1] if body[0] == "block" else [body], ind + 1)
            out.append(p + "}")
            continue
        raise SyntaxError(f"unhandled {s}")
    return out


def lookahead(parser_src):
    methods = find_methods(parser_src)
    fns = {k: v for k, v in methods.items() if re.match(r"^jj_3", k)}
    out = []
    # jj_la1 masks
    for word in ("0", "1", "2"):
        m = re.search(r"jj_la1_" + word + r"\s*=\s*new int\[\]\s*\{(.*?)\};", parser_src, re.S)
        if m:
            vals = [x.strip() for x in m.group(1).split(",") if x.strip()]
            out.append(f"/// `jj_la1_{word}`.")
            out.append(f"pub(super) static JJ_LA1_{word}: [u32; {len(vals)}] = [{', '.join(vals)}];")
            out.append("")
    n_la = len([k for k in methods if re.match(r"^jj_2_\d+$", k)])
    out.append(f"/// The number of `jj_2_<n>` lookaheads.")
    out.append(f"pub(super) const NUM_LOOKAHEADS: usize = {n_la};")
    out.append("")
    out.append("/// `jj_3_<n + 1>()`: `Ok(true)` when the lookahead fails.")
    out.append("pub(super) fn jj_3(rt: &mut Runtime<TokenManager>, n: usize) -> Result<bool, Stop> {")
    out.append("    match n {")
    for k in range(1, n_la + 1):
        target = "_" if k == n_la else str(k - 1)
        out.append(f"        {target} => jj_3_{k}(rt),")
    out.append("    }")
    out.append("}")
    def order(k):
        m = re.match(r"^jj_3_(\d+)$", k)
        return (0, int(m.group(1)), "") if m else (1, 0, k)

    for name in sorted(fns, key=order):
        _, _, body = fns[name]
        ss = Parser(tokenize(strip_comments(body))).block()
        # `while (true) { ... }` arrives as an `if`-less statement list
        fname = name if re.match(r"^jj_3_\d+$", name) else snake(name)
        out.append("")
        out.append(f"/// `{name}()`.")
        out.append(f"fn {fname}(rt: &mut Runtime<TokenManager>) -> Result<bool, Stop> {{")
        out += stmts(ss, 1, last=True)
        out.append("}")
    return "\n".join(out)


def main():
    csrc = strip_comments(open(sys.argv[1], encoding="utf-8").read())
    psrc = strip_comments(open(sys.argv[2], encoding="utf-8").read())
    mode = sys.argv[3]
    if mode == "constants":
        kinds, images = constants(csrc)
        out = []
        for n, v in kinds:
            # token kinds only: not the `_`-private ones, nor the lexical
            # states (`Function`, `Range`, `DEFAULT` after `tokenImage`'s kinds)
            if not n.startswith("_") and n == n.upper() and n != "DEFAULT":
                out.append(f"pub(crate) const {n}: i32 = {v};")
        out.append("")
        out.append("/// `tokenImage`.")
        out.append(f"pub(crate) static TOKEN_IMAGE: [&str; {len(images)}] = [")
        for im in images:
            out.append(f'    "{im}",')
        out.append("];")
        print("\n".join(out))
    else:
        print(lookahead(psrc))


if __name__ == "__main__":
    main()
