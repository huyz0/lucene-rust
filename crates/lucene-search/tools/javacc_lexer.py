#!/usr/bin/env python3
"""Translate a JavaCC 7 generated token manager (`*TokenManager.java`) into a
Rust lexer on `crates/lucene-search/src/queryparser/javacc.rs`'s runtime.

The classic and surround lexers were ported by hand, state for state; this
does the same translation mechanically, for a grammar too large to hand-port
reliably (the flexible query parser's three lexical states). Every generated
function keeps its shape: one match arm per `case`, the NFA's state switch
per character class, the string-literal DFA chain. What is left out:

- `getNextToken`, `jjFillToken`, `jjStopAtPos` and the round bookkeeping,
  which the runtime (`NfaCore`, `Dfa::next_token_loop`) already carries;
- NFA arms of states nothing can enter (no start state, no transition, no
  `jjStartNfaWithStates`), and functions and tables nothing calls.

Usage:
    javacc_lexer.py INPUT.java HEADER.rs NEXT_STATES_NAME > OUT.rs

HEADER.rs is copied verbatim to the top of the output (module docs and
imports). The output still wants `cargo fmt`.
"""
import re
import sys

# ----------------------------------------------------------------- tokenize

TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
   |(?P<str>"(?:\\.|[^"\\])*")
   |(?P<chr>'(?:\\.|[^'\\])')
   |(?P<num>0[xX][0-9a-fA-F]+[lL]?|\d+[lL]?)
   |(?P<id>[A-Za-z_$][A-Za-z0-9_$]*)
   |(?P<op>>>>=|<<=|>>=|>>>|<<|>>|==|!=|<=|>=|&&|\|\||&=|\|=|\+\+|--|\+=|-=|[{}()\[\];,.:?~!%^&*|+\-=<>/])
    """,
    re.X,
)


def strip_comments(src):
    out = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        elif src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
        elif src.startswith("/*", i):
            i = src.index("*/", i) + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tokenize(src):
    toks = []
    pos = 0
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m:
            raise SyntaxError(f"cannot tokenize at {src[pos:pos+40]!r}")
        pos = m.end()
        if m.lastgroup != "ws":
            toks.append(m.group())
    return toks


# -------------------------------------------------------------- statements


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self, k=0):
        return self.t[self.i + k] if self.i + k < len(self.t) else None

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, tok):
        got = self.next()
        if got != tok:
            raise SyntaxError(f"expected {tok!r}, got {got!r} at {self.t[self.i-5:self.i+5]}")

    def paren_expr(self):
        """`( ... )`: the tokens inside the balanced parentheses."""
        self.expect("(")
        depth = 1
        out = []
        while True:
            tok = self.next()
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth -= 1
                if depth == 0:
                    return out
            out.append(tok)

    def until(self, stop):
        out = []
        depth = 0
        while True:
            tok = self.peek()
            if depth == 0 and tok == stop:
                self.next()
                return out
            if tok in ("(", "["):
                depth += 1
            elif tok in (")", "]"):
                depth -= 1
            out.append(self.next())

    def block(self):
        self.expect("{")
        stmts = []
        while self.peek() != "}":
            stmts.append(self.stmt())
        self.next()
        return stmts

    def stmt(self):
        tok = self.peek()
        if tok == "{":
            return ("block", self.block())
        if tok == "if":
            self.next()
            cond = self.paren_expr()
            then = self.stmt()
            els = None
            if self.peek() == "else":
                self.next()
                els = self.stmt()
            return ("if", cond, then, els)
        if tok == "switch":
            self.next()
            scrut = self.paren_expr()
            self.expect("{")
            arms = []
            labels = []
            body = []
            while True:
                tok = self.peek()
                if tok in ("case", "default", "}"):
                    if body:
                        arms.append((labels, body))
                        labels, body = [], []
                    if tok == "}":
                        self.next()
                        break
                    self.next()
                    if tok == "case":
                        labels.append(self.until(":"))
                    else:
                        self.expect(":")
                        labels.append(None)
                    continue
                body.append(self.stmt())
            if labels:  # trailing labels with no statements
                arms.append((labels, []))
            return ("switch", scrut, arms)
        if tok == "return":
            self.next()
            return ("return", self.until(";"))
        if tok == "break":
            self.next()
            if self.peek() != ";":
                self.next()  # a label
            self.expect(";")
            return ("break",)
        if tok == "continue":
            self.next()
            self.until(";")
            return ("continue",)
        if tok == "try":
            self.next()
            body = self.block()
            self.expect("catch")
            self.paren_expr()
            handler = self.block()
            return ("try", body, handler)
        if tok == "do":
            self.next()
            body = self.stmt()
            self.expect("while")
            cond = self.paren_expr()
            self.expect(";")
            return ("do", body, cond)
        if tok == "for":
            self.next()
            self.paren_expr()
            return ("loop", self.stmt())
        if tok == "while":  # only `while (true)` is generated
            self.next()
            self.paren_expr()
            return ("loop", self.stmt())
        if tok in ("int", "long", "boolean", "final") and self.peek(1) not in ("(", "=", "["):
            while self.peek() in ("final",):
                self.next()
            ty = self.next()
            name = self.next()
            init = None
            if self.peek() == "=":
                self.next()
                init = self.until(";")
            else:
                self.expect(";")
            return ("decl", ty, name, init)
        return ("expr", self.until(";"))


# -------------------------------------------------------------- expressions

METHOD_MAP = [
    (re.compile(r"^jjStopStringLiteralDfa_(\d+)$"), "self.stop_string_literal_dfa_{}"),
    (re.compile(r"^jjStartNfaWithStates_(\d+)$"), "self.start_nfa_with_states_{}"),
    (re.compile(r"^jjStartNfa_(\d+)$"), "self.start_nfa_{}"),
    (re.compile(r"^jjMoveStringLiteralDfa(\d+)_(\d+)$"), "self.move_string_literal_dfa{}_{}"),
    (re.compile(r"^jjMoveNfa_(\d+)$"), "self.move_nfa_{}"),
    (re.compile(r"^jjCanMove_(\d+)$"), "can_move_{}"),
    (re.compile(r"^jjbitVec(\d+)$"), "JJ_BIT_VEC{}"),
]
SIMPLE_MAP = {
    "jjStopAtPos": "self.stop_at_pos",
    "jjCheckNAdd": "self.check_n_add",
    "jjCheckNAddTwoStates": "self.two",
    "jjCheckNAddStates": "self.states",
    "jjAddStates": "self.add_states",
    "jjmatchedKind": "self.jjmatched_kind",
    "jjmatchedPos": "self.jjmatched_pos",
    "hiByte": "hi_byte",
    "startState": "start_state",
    "curPos": "cur_pos",
}


def map_id(tok, cur):
    if tok == "curChar":
        return cur
    if tok in SIMPLE_MAP:
        return SIMPLE_MAP[tok]
    for rx, fmt in METHOD_MAP:
        m = rx.match(tok)
        if m:
            return fmt.format(*m.groups())
    return tok


def map_num(tok):
    t = tok.rstrip("lL")
    if t.lower().startswith("0x"):
        return "0x" + t[2:].lower()
    if len(t) > 1 and t.startswith("0"):
        return str(int(t, 8))  # Java octal
    return t


def expr(toks, cur):
    """Java expression tokens -> Rust text (spacing left to rustfmt)."""
    out = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        # `(int)` / `(long)` casts
        if tok == "(" and i + 2 < len(toks) and toks[i + 1] in ("int", "long") and toks[i + 2] == ")":
            i += 3
            continue
        if re.match(r"^\d|^0[xX]", tok):
            out.append(map_num(tok))
        elif re.match(r"^[A-Za-z_$]", tok):
            out.append(map_id(tok, cur))
        else:
            out.append(tok)
        i += 1
    s = ""
    for tok in out:
        if s and not (s.endswith("(") or s.endswith("[") or tok in (")", "]", ",", "(", "[")) or (tok == "(" and s and s[-1].isalnum() is False and s[-1] not in "_"):
            s += " "
        s += tok
    s = re.sub(r"(\w) \(", r"\1(", s)
    s = re.sub(r"(\w) \[", r"\1[", s)
    s = s.replace("! ", "!")
    return s


def outer(toks):
    """Drop parentheses that wrap the whole expression."""
    while toks and toks[0] == "(" and toks[-1] == ")":
        depth = 0
        for k, tok in enumerate(toks):
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth -= 1
                if depth == 0 and k != len(toks) - 1:
                    return toks
        toks = toks[1:-1]
    return toks


# ------------------------------------------------------------------ emitter


class Emitter:
    def __init__(self, cur, label=None, nfa=False):
        self.cur = cur
        self.label = label
        self.nfa = nfa
        self.used_label = False

    def stmts(self, stmts, ind):
        lines = []
        for s in stmts:
            lines += self.stmt(s, ind)
        return lines

    def body(self, s, ind):
        if s[0] == "block":
            return self.stmts(s[1], ind)
        return self.stmt(s, ind)

    def stmt(self, s, ind):
        p = "    " * ind
        kind = s[0]
        if kind == "block":
            return [p + "{"] + self.stmts(s[1], ind + 1) + [p + "}"]
        if kind == "if":
            _, cond, then, els = s
            cond = outer(cond)
            # `if (kind > N) kind = N;`
            if (
                len(cond) == 3 and cond[0] == "kind" and cond[1] == ">"
                and then[0] == "expr" and then[1] == ["kind", "=", cond[2]] and els is None
            ):
                return [p + f"kind = kind.min({map_num(cond[2])});"]
            # `if (((active0 &= old0)) == 0L) ...`
            flat = [t for t in cond if t not in ("(", ")")]
            pre = []
            if "&=" in flat:
                k = flat.index("&=")
                pre = [p + f"{flat[k-1]} &= {flat[k+1]};"]
                cond = [flat[k - 1]] + flat[k + 2 :]
            lines = pre + [p + f"if {expr(cond, self.cur)} {{"] + self.body(then, ind + 1)
            if els is None:
                return lines + [p + "}"]
            if els[0] == "if":
                rest = self.stmt(els, ind)
                return lines + [p + "} else " + rest[0].lstrip()] + rest[1:]
            return lines + [p + "} else {"] + self.body(els, ind + 1) + [p + "}"]
        if kind == "return":
            text = expr(outer(s[1]), self.cur)
            # `jjMoveNfa_N(jjStopStringLiteralDfa_N(..), ..)`: two borrows of
            # `self` in one call, so the inner call goes first
            m = re.match(r"^(self\.move_nfa_\d+)\((self\.stop_string_literal_dfa_\d+\([^()]*\)), (.*)\)$", text)
            if m:
                return [p + f"let state = {m.group(2)};", p + f"return {m.group(1)}(state, {m.group(3)});"]
            return [p + f"return {text};"]
        if kind == "break":
            if self.label is None:
                raise SyntaxError("break outside a switch arm")
            self.used_label = True
            return [p + f"break '{self.label};"]
        if kind == "try":
            _, body, handler = s
            if not (len(body) == 1 and body[0][0] == "expr" and "readChar" in body[0][1]):
                raise SyntaxError(f"unexpected try: {body}")
            return [p + "if !self.read() {"] + self.stmts(handler, ind + 1) + [p + "}"]
        if kind == "switch":
            return self.switch(s, ind)
        if kind == "expr":
            toks = s[1]
            # `jjstateSet[jjnewStateCnt++] = N;`
            if toks[:5] == ["jjstateSet", "[", "jjnewStateCnt", "++", "]"]:
                return [p + f"self.push({expr(toks[6:], self.cur)});"]
            return [p + expr(toks, self.cur) + ";"]
        if kind == "decl":
            _, ty, name, init = s
            return [p + f"let {map_id(name, self.cur)} = {expr(outer(init), self.cur)};"]
        raise SyntaxError(f"unhandled statement {s}")

    def switch(self, s, ind):
        _, scrut, arms = s
        p = "    " * ind
        lines = [p + f"match {expr(outer(scrut), self.cur)} {{"]
        default = None
        for labels, body in arms:
            body = list(body)
            if body and body[-1] == ("break",):
                body = body[:-1]
            pats = [map_num(l[0]) if len(l) == 1 else expr(l, self.cur) for l in labels if l is not None]
            if None in labels:
                default = body
                continue
            lines += self.arm(" | ".join(pats), body, ind + 1)
        lines += self.arm("_", default or [], ind + 1)
        return lines + [p + "}"]

    def arm(self, pat, body, ind):
        p = "    " * ind
        inner = Emitter(self.cur, label="arm", nfa=self.nfa)
        content = inner.stmts(body, ind + 2 if self._needs_label(body) else ind + 1)
        if not content:
            return [p + f"{pat} => {{}}"]
        if inner.used_label:
            return [p + f"{pat} => {{", p + "    'arm: {"] + content + [p + "    }", p + "}"]
        return [p + f"{pat} => {{"] + content + [p + "}"]

    @staticmethod
    def _needs_label(body):
        def has_break(s):
            if s[0] == "break":
                return True
            if s[0] == "block":
                return any(has_break(x) for x in s[1])
            if s[0] == "if":
                return has_break(s[2]) or (s[3] is not None and has_break(s[3]))
            return False

        return any(has_break(s) for s in body)


# --------------------------------------------------------------- the lexer


def find_methods(src):
    head = re.compile(
        r"(?:private|public|protected)?\s*(?:static\s+)?(?:final\s+)?(int|boolean|void|Token)\s+(\w+)\s*\(([^)]*)\)\s*\{"
    )
    methods = {}
    for m in head.finditer(src):
        start = m.end() - 1
        depth = 0
        i = start
        while True:
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            elif c in "\"'":
                q = c
                i += 1
                while src[i] != q:
                    i += 2 if src[i] == "\\" else 1
            i += 1
        methods[m.group(2)] = (m.group(1), m.group(3), src[start : i + 1])
    return methods


def find_array(src, name):
    m = re.search(r"\b" + name + r"\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        return None
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def java_string(lit):
    """A Java string literal with octal escapes -> its text."""
    body = lit[1:-1]
    return re.sub(r"\\([0-7]{1,3})", lambda m: chr(int(m.group(1), 8)), body).replace("\\\\", "\\").replace('\\"', '"')


def calls(text):
    return set(re.findall(r"\b(jj\w+)\s*\(", text))


def nfa_targets(stmts, next_states):
    """Every state an NFA arm's statements can add to the next state set."""
    out = set()

    def walk(s):
        if s[0] in ("block",):
            for x in s[1]:
                walk(x)
        elif s[0] == "if":
            walk(s[2])
            if s[3] is not None:
                walk(s[3])
        elif s[0] == "expr":
            toks = s[1]
            if toks[:5] == ["jjstateSet", "[", "jjnewStateCnt", "++", "]"]:
                out.add(int(map_num(toks[6])))
            elif toks and toks[0] in ("jjCheckNAdd",):
                out.add(int(map_num(toks[2])))
            elif toks and toks[0] == "jjCheckNAddTwoStates":
                out.add(int(map_num(toks[2])))
                out.add(int(map_num(toks[4])))
            elif toks and toks[0] in ("jjCheckNAddStates", "jjAddStates"):
                a = int(map_num(toks[2]))
                b = int(map_num(toks[4])) if len(toks) > 4 and toks[3] == "," else a
                out.update(next_states[a : b + 1])

    for s in stmts:
        walk(s)
    return out


def translate_nfa(name, body_src, next_states, starts):
    """A `jjMoveNfa_N`: its size and the three character classes' switches,
    arms of unreachable states dropped."""
    n = name.split("_")[1]
    p = Parser(tokenize(body_src))
    stmts = p.block()
    size = None
    loop = None
    for s in stmts:
        if s[0] == "expr" and s[1][:2] == ["jjnewStateCnt", "="]:
            size = int(s[1][2])
        if s[0] == "loop":
            loop = s[1]
    branches = None
    for s in loop[1]:
        if s[0] == "if" and outer(s[1]) == ["curChar", "<", "64"]:
            branches = s
    lt64 = branches[2]
    lt128 = branches[3][2]
    rest = branches[3][3]

    def switch_of(block):
        for s in block[1]:
            if s[0] == "do":
                for x in s[1][1]:
                    if x[0] == "switch":
                        return x
        raise SyntaxError("no switch")

    sections = [switch_of(lt64), switch_of(lt128), switch_of(rest)]
    # reachability over the three sections' transitions
    reach = set(starts)
    edges = {}
    for sw in sections:
        for labels, body in sw[2]:
            targets = nfa_targets(body, next_states)
            for l in labels:
                if l is not None:
                    edges.setdefault(int(map_num(l[0])), set()).update(targets)
    frontier = list(reach)
    while frontier:
        st = frontier.pop()
        for t in edges.get(st, ()):
            if t not in reach:
                reach.add(t)
                frontier.append(t)

    def prune(sw):
        arms = []
        for labels, body in sw[2]:
            kept = [l for l in labels if l is None or int(map_num(l[0])) in reach]
            if any(l is not None for l in kept) or None in kept:
                arms.append((kept, body))
        return ("switch", sw[1], arms)

    em = Emitter("c", nfa=True)
    out = []
    out.append(f"    fn move_nfa_{n}(&mut self, start_state: i32, mut cur_pos: i32) -> i32 {{")
    out.append("        let mut starts_at = 0usize;")
    out.append(f"        self.nfa_start(start_state, {size});")
    out.append("        let mut i = 1usize;")
    out.append("        let mut kind = NO_KIND;")
    out.append("        loop {")
    out.append("            self.next_round();")
    out.append("            let c = self.cur_char;")
    heads = [
        ("if c < 64 {", ["let l = 1u64 << c;"]),
        ("} else if c < 128 {", ["let l = 1u64 << (c & 63);"]),
        ("} else {", [
            "let hi_byte = c >> 8;",
            "let i1 = (hi_byte >> 6) as usize;",
            "let l1 = 1u64 << (hi_byte & 63);",
            "let i2 = ((c & 0xff) >> 6) as usize;",
            "let l2 = 1u64 << (c & 63);",
        ]),
    ]
    for (head, decls), sw in zip(heads, sections):
        sw = prune(sw)
        out.append("            " + head)
        real = [a for a in sw[2] if any(l is not None for l in a[0])]
        if not real:
            out.append("                // no state moves on these characters")
            out.append("                i = starts_at;")
            continue
        used = stmt_tokens([s for a in real for s in a[1]])
        canmove = any(t.startswith("jjCanMove") for t in used)
        for d in decls:
            var = d.split()[1]
            if var in used or (var in ("i1", "l1", "i2", "l2", "hi_byte") and canmove):
                out.append("                " + d)
        out.append("                loop {")
        out.append("                    i -= 1;")
        out += em.switch(("switch", ["jjstateSet_i"], real), 5)
        out.append("                    if i == starts_at {")
        out.append("                        break;")
        out.append("                    }")
        out.append("                }")
    out.append("            }")
    out.append(
        f"            if let Err(pos) = self.nfa_step(&mut kind, &mut cur_pos, &mut i, &mut starts_at, {size}) {{"
    )
    out.append("                return pos;")
    out.append("            }")
    out.append("        }")
    out.append("    }")
    text = "\n".join(out).replace("match jjstateSet_i {", "match self.state_at(i) {")
    return text, size


def stmt_tokens(stmts):
    """Every token of the statements, nested ones included."""
    out = set()

    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, str):
            out.add(x)

    walk(stmts)
    return out


def rust_str(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def rust_params(params, mutate):
    out = []
    for prm in [x.strip() for x in params.split(",") if x.strip()]:
        ty, nm = prm.split()
        rty = {"int": "i32", "long": "u64"}[ty]
        nm = map_id(nm, "c")
        if nm in ("hi_byte",):
            pass
        out.append(("mut " if nm in mutate else "") + f"{nm}: {rty}")
    return ", ".join(out)


def main():
    src = strip_comments(open(sys.argv[1], encoding="utf-8").read())
    header = open(sys.argv[2], encoding="utf-8").read()
    methods = find_methods(src)
    next_states = [int(x) for x in find_array(src, "jjnextStates")]

    # the lexical states' entry points, from getNextToken's switch
    gnt = methods["getNextToken"][2]
    entries = dict(re.findall(r"case\s+(\d+):\s*jjmatchedKind\s*=\s*0x7fffffff;\s*jjmatchedPos\s*=\s*0;\s*curPos\s*=\s*(\w+)\(\);", gnt))

    # what is called from the entry points, transitively
    funcs = {k: v for k, v in methods.items() if re.match(r"jj(StopStringLiteralDfa|StartNfa|MoveStringLiteralDfa|StartNfaWithStates|MoveNfa|CanMove)", k)}
    live = set()
    frontier = list(entries.values())
    while frontier:
        f = frontier.pop()
        if f in live or f not in funcs:
            continue
        live.add(f)
        frontier += [c for c in calls(funcs[f][2]) if c in funcs]

    # an NFA's start states: constants passed to it, a stop DFA's returns,
    # and the states jjStartNfaWithStates hands it
    starts = {}
    for f in live:
        body = funcs[f][2]
        for nfa, st in re.findall(r"jjMoveNfa_(\d+)\((\d+),", body):
            starts.setdefault(nfa, set()).add(int(st))
        for nfa, st in re.findall(r"jjStartNfaWithStates_(\d+)\(\d+,\s*\d+,\s*(\d+)\)", body):
            starts.setdefault(nfa, set()).add(int(st))
        m = re.match(r"jjStopStringLiteralDfa_(\d+)", f)
        if m:
            for st in re.findall(r"return\s+(\d+);", body):
                starts.setdefault(m.group(1), set()).add(int(st))
        m = re.match(r"jjStartNfaWithStates_(\d+)", f)

    out = [header.rstrip("\n"), ""]
    # tables
    used_vecs = set()
    for f in live:
        used_vecs |= set(re.findall(r"jjbitVec(\d+)", funcs[f][2]))
    for v in sorted(used_vecs, key=int):
        vals = ", ".join(map_num(x) for x in find_array(src, f"jjbitVec{v}"))
        out.append(f"static JJ_BIT_VEC{v}: [u64; 4] = [{vals}];")
    out.append("")
    out.append(f"/// `jjnextStates`.\nstatic JJ_NEXT_STATES: [i32; {len(next_states)}] = [{', '.join(map(str, next_states))}];")
    out.append("")
    # jjCanMove_N (free functions)
    for f in sorted(x for x in live if x.startswith("jjCanMove")):
        ret, params, body = funcs[f]
        stmts = Parser(tokenize(body)).block()
        names = [p.split()[1] for p in params.split(",")]
        body_ids = set(tokenize(body))
        prm = []
        for p in params.split(","):
            ty, nm = p.split()
            rnm = map_id(nm, "c")
            rty = "usize" if nm in ("i1", "i2") else ("u64" if ty == "long" else "i32")
            prm.append(("" if nm in body_ids else "_") + f"{rnm}: {rty}")
        n = f.split("_")[1]
        out.append(f"/// `jjCanMove_{n}`.")
        out.append(f"fn can_move_{n}({', '.join(prm)}) -> bool {{")
        out += Emitter("c").stmts(stmts, 1)
        out.append("}")
        out.append("")

    out.append("pub(crate) struct TokenManager {\n    core: NfaCore,\n}\n")
    out.append("impl Deref for TokenManager {\n    type Target = NfaCore;\n    fn deref(&self) -> &NfaCore {\n        &self.core\n    }\n}\n")
    out.append("impl DerefMut for TokenManager {\n    fn deref_mut(&mut self) -> &mut NfaCore {\n        &mut self.core\n    }\n}\n")
    rounds = int(re.search(r"jjrounds\s*=\s*new int\[(\d+)\]", src).group(1))
    default_state = re.search(r"int\s+curLexState\s*=\s*(\d+);", src)
    default_state = int(default_state.group(1)) if default_state else 0
    out.append("impl TokenManager {")
    out.append("    pub(crate) fn new(input: CharStream) -> Self {")
    out.append("        TokenManager {")
    out.append(f"            core: NfaCore::new(input, {rounds}, &JJ_NEXT_STATES, {default_state}),")
    out.append("        }")
    out.append("    }")

    def key(f):
        m = re.search(r"_(\d+)$", f)
        order = ["jjStopStringLiteralDfa", "jjStartNfa_", "jjMoveStringLiteralDfa", "jjStartNfaWithStates", "jjMoveNfa"]
        rank = next((k for k, o in enumerate(order) if f.startswith(o)), 9)
        lvl = re.search(r"Dfa(\d+)_", f)
        return (int(m.group(1)) if m else 0, rank, int(lvl.group(1)) if lvl else 0)

    for f in sorted((x for x in live if not x.startswith("jjCanMove")), key=key):
        ret, params, body = funcs[f]
        out.append("")
        if f.startswith("jjMoveNfa"):
            n = f.split("_")[1]
            text, _ = translate_nfa(f, body, next_states, starts.get(n, set()))
            out.append(text)
            continue
        mutate = {"active0"} if "&=" in body else set()
        rname = map_id(f, "c").replace("self.", "")
        if f.startswith("jjStopStringLiteralDfa"):
            out.append("    // SENTINEL: `-1` = no state: the NFA it starts matches nothing.")
        out.append(f"    fn {rname}(&mut self, {rust_params(params, mutate)}) -> i32 {{")
        stmts = Parser(tokenize(body)).block()
        out += Emitter("self.cur_char").stmts(stmts, 2)
        out.append("    }")
    out.append("}")
    out.append("")

    # Dfa
    images = find_array(src, "jjstrLiteralImages")
    new_lex = [int(x) for x in find_array(src, "jjnewLexState")]
    to_token = [map_num(x) for x in find_array(src, "jjtoToken")]
    out.append("impl Dfa for TokenManager {")
    out.append("    fn dispatch(&mut self) -> i32 {")
    out.append("        match self.cur_lex_state {")
    for state, fn in sorted(entries.items(), key=lambda kv: int(kv[0])):
        if int(state) != default_state:
            out.append(f"            {state} => {map_id(fn, 'c')}(),")
    out.append(f"            _ => {map_id(entries[str(default_state)], 'c')}(),")
    out.append("        }")
    out.append("    }")
    out.append("")
    out.append("    /// `jjnewLexState`.")
    out.append("    // SENTINEL: `-1` = the kind does not change the lexical state.")
    out.append("    fn new_lex_state(kind: i32) -> i32 {")
    out.append("        match kind {")
    groups = {}
    for k, v in enumerate(new_lex):
        if v != -1:
            groups.setdefault(v, []).append(k)
    for v, ks in sorted(groups.items()):
        out.append(f"            {' | '.join(map(str, ks))} => {v},")
    out.append("            _ => -1,")
    out.append("        }")
    out.append("    }")
    out.append("")
    out.append(f"    /// `jjtoToken` (`{', '.join(to_token)}`).")
    out.append("    fn is_token(kind: i32) -> bool {")
    words = ", ".join(to_token)
    out.append(f"        const JJ_TO_TOKEN: [u64; {len(to_token)}] = [{words}];")
    out.append("        let k = kind as usize;")
    out.append("        JJ_TO_TOKEN[k >> 6] & (1u64 << (k & 63)) != 0")
    out.append("    }")
    out.append("")
    out.append("    /// `jjstrLiteralImages`.")
    out.append("    fn literal_image(kind: i32) -> Option<&'static str> {")
    out.append("        Some(match kind {")
    for k, lit in enumerate(images):
        if lit != "null":
            text = java_string(lit)
            out.append(f"            {k} => {rust_str(text)},")
    out.append("            _ => return None,")
    out.append("        })")
    out.append("    }")
    out.append("}")
    out.append("")
    out.append("impl Lexer for TokenManager {")
    out.append("    fn next_token(&mut self) -> Result<Token, TokenMgrError> {")
    out.append("        self.next_token_loop()")
    out.append("    }")
    out.append("}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
