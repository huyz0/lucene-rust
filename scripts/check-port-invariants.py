#!/usr/bin/env python3
"""Mechanical gates for defect shapes this port's M2 sweep kept re-finding by hand.

Companion to `scripts/check-arith-allows.py`. That script enforces the two
*arithmetic* rules in `docs/arithmetic-gate.md`; this one enforces the rules in
that document that `clippy::arithmetic_side_effects` structurally cannot see,
plus two record-keeping rules a Tier-2 review found by eye.

Every rule below exists because the same defect was found **by hand more than
once**, in unrelated modules, after an audit had already walked past it. What
each rule can and cannot catch is written down in `docs/mechanical-gates.md`;
read that before trusting a green run. A gate whose blind spot is undocumented
is worse than no gate, because it is read as coverage.

Run: `python3 scripts/check-port-invariants.py [--verbose]`
Exit 0 = clean, 1 = at least one violation.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATES = os.path.join(ROOT, "crates")


# --------------------------------------------------------------------------
# Shared source handling
# --------------------------------------------------------------------------


def src_files():
    """Every `crates/*/src/**.rs` file, sorted, as (relpath, [lines])."""
    out = []
    for crate in sorted(os.listdir(CRATES)):
        src = os.path.join(CRATES, crate, "src")
        if not os.path.isdir(src):
            continue
        for dirpath, _dirnames, filenames in os.walk(src):
            for name in sorted(filenames):
                if not name.endswith(".rs"):
                    continue
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, ROOT)
                with open(path, encoding="utf-8") as fh:
                    out.append((rel, fh.read().split("\n")))
    return sorted(out)


def blank_cfg_test(lines):
    """Replace every `#[cfg(test)]` item's lines with empty strings.

    Line numbers are preserved so every diagnostic still points at the real
    line. Test code is deliberately out of scope for all of these rules: a
    test's whole job is to build the shapes production code must refuse, and a
    gate that fires inside `mod tests` is one that gets `#[allow]`-ed
    everywhere and stops meaning anything (the same reasoning
    `docs/arithmetic-gate.md` gives for its own test-code carve-out).
    """
    out = list(lines)
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() != "#[cfg(test)]":
            i += 1
            continue
        # The attribute may be followed by more attributes, then an item that
        # is either brace-delimited (`mod tests { .. }`, `impl { .. }`) or a
        # single statement ending in `;` (`use ...;`).
        j = i
        depth = 0
        started = False
        while j < n:
            line = lines[j]
            if "{" in line:
                started = True
            depth += line.count("{") - line.count("}")
            if not started and line.rstrip().endswith(";"):
                break
            if started and depth <= 0:
                break
            j += 1
        for k in range(i, min(j + 1, n)):
            out[k] = ""
        i = j + 1
    return out


FN_START = re.compile(
    r"^\s*(pub(\([\w: ]+\))?\s+)?(default\s+)?(const\s+)?(async\s+)?(unsafe\s+)?"
    r'(extern\s+"[^"]*"\s+)?fn\s+(?P<name>\w+)'
)


def fn_spans(lines):
    """Yield (name, first_line_idx, last_line_idx) for every `fn` in `lines`.

    Brace counting, not a parser: braces inside string literals would throw it
    off. No such `fn` signature exists in this tree, and the failure mode is a
    span that is too long -- which makes a rule *more* permissive, never less,
    so it cannot manufacture a false failure.
    """
    n = len(lines)
    for i, line in enumerate(lines):
        m = FN_START.match(line)
        if not m:
            continue
        j = i
        depth = 0
        started = False
        while j < n:
            if "{" in lines[j]:
                started = True
            depth += lines[j].count("{") - lines[j].count("}")
            if not started and lines[j].rstrip().endswith(";"):
                # A trait method declaration with no body.
                break
            if started and depth <= 0:
                break
            j += 1
        yield m.group("name"), i, min(j, n - 1)


def strip_comment(line):
    """The code half of a line, so a rule cannot be tripped by prose."""
    idx = line.find("//")
    return line if idx < 0 else line[:idx]


def leading_comment(lines, idx, span=6):
    """The comment text on the `span` lines immediately above `idx`."""
    start = max(0, idx - span)
    return "\n".join(lines[start:idx])


# --------------------------------------------------------------------------
# Rule 1: a `FixedBitSet` is only ever indexed by a bound taken from itself
# --------------------------------------------------------------------------
#
# `FixedBitSet::get`/`set`/`clear` index `words[index >> 6]`. An index past
# `num_bits` but inside the final word is a *ghost bit* -- a silently wrong
# live/dead answer that no error path can catch -- and one 64 or more past the
# end is an index panic, in release as well as debug.
#
# Found by hand three times: twice in one crate by c28
# (`term_delete::resolve_term_doc_ids` with no bound at all, `deletes::
# mark_deleted` bounded against a separate `max_doc` parameter), then again by
# c30 in `merge_segments` (the `.liv` indexed by a bound taken off the `.fdm`).
# `clippy::arithmetic_side_effects` sees none of it: this is the *indexing* row
# of the arithmetic gate's table, which that lint does not cover.

BITSET_TYPE = "FixedBitSet"
BITSET_METHODS = ("get", "set", "clear")
FBS_PROOF = "FBS:"


# Methods no other type in this workspace has, so a receiver that calls one is
# a `FixedBitSet` whatever the binding looked like.
BITSET_TELLS = ("cardinality()", "words()", "clear_all()")


def let_statements(text):
    """Every `let ...;` statement in `text`, re-joined across lines.

    `let mut bits = match live_docs { Some(e) => e.clone(), None =>
    FixedBitSet::from_words(..) };` binds a bitset over five lines, and a
    line-at-a-time scan sees none of it -- which is how `deletes::mark_deleted`,
    one of the two sites c28 actually fixed, would have slipped past this rule.
    """
    out = []
    for m in re.finditer(r"\blet\s+(mut\s+)?(\w+)", text):
        name = m.group(2)
        depth = 0
        i = m.end()
        while i < len(text):
            ch = text[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == ";" and depth <= 0:
                break
            i += 1
        out.append((name, text[m.start() : i]))
    return out


def bitset_names(text):
    """Identifiers this file binds to a `FixedBitSet`.

    Deliberately file-wide rather than scope-accurate: a name bound to a
    bitset anywhere in a file is treated as a bitset everywhere in it. That
    over-approximates, which is the safe direction -- it can ask for a bound
    that was not strictly needed (answerable with an `// FBS:` proof), and
    cannot silently drop a real site.
    """
    names = set()
    # Parameters, struct fields and typed lets: `name: FixedBitSet`,
    # `name: &FixedBitSet`, `name: Option<&mut FixedBitSet>`, ...
    for m in re.finditer(
        r"\b(\w+)\s*:\s*(&\s*)?(mut\s+)?(Option\s*<\s*)?(&\s*)?(mut\s+)?" + BITSET_TYPE,
        text,
    ):
        names.add(m.group(1))
    # A receiver calling a `FixedBitSet`-only method.
    for tell in BITSET_TELLS:
        for m in re.finditer(r"([A-Za-z_][\w.]*)\." + re.escape(tell), text):
            names.add(m.group(1).split(".")[-1])
    # `let` statements that mention the constructor, or an already-known
    # bitset, anywhere on their right-hand side. Iterated to a fixpoint so a
    # chain (`let a = FixedBitSet::new(..); let b = a.clone();`) is followed.
    lets = let_statements(text)
    for _ in range(4):
        before = len(names)
        for name, stmt in lets:
            if name in names:
                continue
            if BITSET_TYPE + "::" in stmt or any(
                re.search(r"(?<![\w.])" + re.escape(n) + r"\s*\.\s*(clone|to_owned)\b", stmt)
                for n in names
            ):
                names.add(name)
        if len(names) == before:
            break
    # Rebindings of a known bitset that no `let` and no type annotation names.
    # These are the *dominant* shape in `lucene-search`, and leaving them out
    # made the rule miss roughly half the real index sites -- which c41's own
    # Tier-2 review caught, in a rule written to catch exactly that:
    #   live_docs.is_none_or(|bits| bits.get(doc as usize))
    #   if let Some(bits) = live_docs { .. }
    #   match live_docs { Some(bits) => .. }
    #   for bits in &live_docs_per_segment { .. }
    for _ in range(3):
        before = len(names)
        known = "|".join(re.escape(n) for n in sorted(names)) if names else None
        if not known:
            break
        # A closure whose receiver is a known bitset: `<known>.f(|p| ..)`.
        for m in re.finditer(
            r"(?<![\w.])(?:" + known + r")\s*(?:\.\s*\w+\s*)*\(\s*\|\s*(&?\s*mut\s+)?(\w+)\s*\|",
            text,
        ):
            names.add(m.group(2))
        # `if let Some(p) = <known>` / `while let Some(p) = <known>` /
        # `Some(p) => ..` inside a `match` on a known bitset, and `for p in
        # <known>`.
        for m in re.finditer(
            r"\b(?:if|while)\s+let\s+Some\s*\(\s*(?:ref\s+)?(?:mut\s+)?(\w+)\s*\)"
            r"\s*=\s*(?:&\s*(?:mut\s+)?)?(?:" + known + r")\b",
            text,
        ):
            names.add(m.group(1))
        for m in re.finditer(
            r"\bmatch\s+(?:&\s*(?:mut\s+)?)?(?:" + known + r")\b[^{]*\{([^}]*)\}", text
        ):
            for a in re.finditer(r"Some\s*\(\s*(?:ref\s+)?(?:mut\s+)?(\w+)\s*\)\s*=>", m.group(1)):
                names.add(a.group(1))
        for m in re.finditer(
            r"\bfor\s+(?:&\s*)?(?:mut\s+)?(\w+)\s+in\s+(?:&\s*(?:mut\s+)?)?(?:" + known + r")\b",
            text,
        ):
            names.add(m.group(1))
        if len(names) == before:
            break
    names.discard("")
    return names


CALL_RE = re.compile(r"(?P<recv>[A-Za-z_][\w.]*)\.(?P<method>get|set|clear)\s*\(")


def rule_fixed_bitset_bound(files, problems, stats):
    for rel, raw in files:
        text = "\n".join(raw)
        if BITSET_TYPE not in text:
            continue
        lines = blank_cfg_test(raw)
        names = bitset_names(text)
        if not names:
            continue
        for _fname, a, b in fn_spans(lines):
            # Code only: a `bits.len()` written inside a *comment* must not
            # waive the rule for the whole function.
            body = "\n".join(strip_comment(line) for line in lines[a : b + 1])
            for k in range(a, b + 1):
                code = strip_comment(lines[k])
                for m in CALL_RE.finditer(code):
                    recv = m.group("recv")
                    if recv.split(".")[-1] not in names:
                        continue
                    stats["fbs_sites"] += 1
                    if f"{recv}.len()" in body or f"{recv}.is_empty()" in body:
                        continue
                    # A proof is prose and routinely runs to several lines, so
                    # the window is wider than a one-line `// SAFETY:` would
                    # need.
                    if FBS_PROOF in leading_comment(lines, k, span=14):
                        stats["fbs_proofs"] += 1
                        continue
                    problems.append(
                        f"{rel}:{k + 1}: `{recv}.{m.group('method')}(..)` indexes a "
                        f"FixedBitSet, but nothing in the enclosing fn takes "
                        f"`{recv}.len()`. Bound the index against the bitset's own "
                        f"length, or justify it with an `// FBS:` comment. "
                        f"(docs/mechanical-gates.md#fixed-bitset-bound)"
                    )


# --------------------------------------------------------------------------
# Rule 2: an out-of-domain sentinel is declared, and checked at every call site
# --------------------------------------------------------------------------
#
# A function returning a sentinel *outside* the domain of its result (`-1` for
# "no next set bit", "not found", ...) has to be checked by every caller, and
# an audit that records "this function's sentinel is handled" instead of "this
# call site handles it" will miss one. c31 shipped exactly that: it bounded
# `bit_table_next_bit_set`'s upper end and left the `-1` unchecked one function
# over, where it became an arc one label below the node's declared range -- and
# for `firstLabel == 0`, exactly `END_LABEL`.
#
# A byte-flip sweep structurally cannot find this: it asserts "a typed error or
# a clean decode", and a plausible wrong label *is* a clean decode. c31's sweep
# ran 40 136 flips over this code and passed.

SENTINEL_DECL = "SENTINEL:"
SENTINEL_OK = "SENTINEL-OK:"
# How far after a call the sentinel test may sit. Wide enough for the prose
# these checks carry (the c31 one runs to twelve comment lines), narrow enough
# that an unrelated `< 0` further down the function cannot silently satisfy it.
SENTINEL_WINDOW = 22
INT_RET = re.compile(r"^(i8|i16|i32|i64|isize)$")
WRAPPED_RET = re.compile(r"^(Result|Option)\s*<\s*([^<>,]+?)\s*>$")
# A body that hands `-1` back to the caller, in any of the spellings this tree
# uses: `return -1`, `return Ok(-1)`, a match arm `=> -1`, or a tail `-1`.
RETURNS_MINUS_ONE = re.compile(
    r"return\s+(Ok\(\s*)?-1\b|=>\s*(Ok\(\s*)?-1\b|^\s*(Ok\(\s*)?-1\s*\)?,?\s*$",
    re.M,
)
# Anything that constitutes testing the sentinel at a call site.
SENTINEL_CHECKED = re.compile(
    r"==\s*-1|!=\s*-1|<\s*0\b|>=\s*0\b|>\s*-1\b|\bis_negative\(\)|"
    r"\bu32::try_from|\busize::try_from|\bu64::try_from|\.max\(0\)|"
    r"NO_MORE_DOCS|\bmatches!\s*\(\s*\w+\s*,\s*-1"
)


def declared_sentinels(files):
    """Functions whose doc/comment block declares an out-of-domain sentinel."""
    declared = {}
    undeclared = []
    for rel, raw in files:
        lines = blank_cfg_test(raw)
        for name, a, b in fn_spans(lines):
            sig = " ".join(lines[a : min(a + 6, b + 1)])
            m = re.search(r"->\s*(.+?)\s*\{", sig)
            if not m:
                continue
            ret = m.group(1).strip()
            core = ret
            wm = WRAPPED_RET.match(core)
            if wm:
                core = wm.group(2).strip()
            if not INT_RET.match(core):
                continue
            body = "\n".join(lines[a : b + 1])
            if not RETURNS_MINUS_ONE.search(body):
                continue
            if SENTINEL_DECL in leading_comment(lines, a, span=30):
                declared[name] = (rel, a + 1)
            else:
                undeclared.append((rel, a + 1, name, ret))
    return declared, undeclared


def rule_sentinel_callers(files, problems, stats):
    declared, undeclared = declared_sentinels(files)
    for rel, lineno, name, ret in undeclared:
        problems.append(
            f"{rel}:{lineno}: `fn {name}(..) -> {ret}` returns a bare `-1` "
            f"sentinel with no `// SENTINEL:` declaration. Declare what the "
            f"sentinel means, so every call site can be checked. "
            f"(docs/mechanical-gates.md#sentinel-callers)"
        )
    for name, (decl_rel, decl_line) in sorted(declared.items()):
        decl_mod = os.path.basename(decl_rel)[: -len(".rs")]
        # A bare `name(` or one qualified with the *declaring* module. A call
        # qualified with any other path is a different function that happens to
        # share the name (`doc_score_encoder::doc_id` vs `postings::doc_id`).
        call_re = re.compile(
            r"(?:(?<![\w:.])|(?<=\b" + re.escape(decl_mod) + r"::))"
            + re.escape(name)
            + r"\s*\("
        )
        for rel, raw in files:
            lines = blank_cfg_test(raw)
            for k, line in enumerate(lines):
                if rel == decl_rel and k + 1 == decl_line:
                    continue
                code = strip_comment(line)
                if not call_re.search(code):
                    continue
                if FN_START.match(line):  # the definition itself
                    continue
                stats["sentinel_sites"] += 1
                window = lines[k : min(k + SENTINEL_WINDOW, len(lines))]
                if any(SENTINEL_CHECKED.search(strip_comment(w)) for w in window):
                    continue
                # The justification may sit above the call or between it and
                # the branch it feeds, so both sides of the call count.
                context = leading_comment(lines, k, span=6) + "\n".join(window)
                if SENTINEL_OK in context:
                    stats["sentinel_waived"] += 1
                    continue
                problems.append(
                    f"{rel}:{k + 1}: call of `{name}` (declared at "
                    f"{decl_rel}:{decl_line} as returning a `-1` sentinel) with no "
                    f"test of the sentinel in the {SENTINEL_WINDOW} lines that "
                    f"follow, and no `// SENTINEL-OK:` justification. "
                    f"(docs/mechanical-gates.md#sentinel-callers)"
                )


# --------------------------------------------------------------------------
# Rule 3: the per-field codec suffix is derived, never spelled out
# --------------------------------------------------------------------------
#
# c14 shipped a hardcoded `"Lucene90_0"` per-field suffix (its F-12). The
# suffix belongs to `index_writer::per_field_codec_suffix` and nowhere else:
# a literal anywhere in production code is a second, silently diverging copy
# of a rule that has to match what `PerFieldDocValuesFormat` computes.

SUFFIX_LITERAL = re.compile(r'"[^"]*Lucene\d{2}_\d')
SUFFIX_OWNER = "crates/lucene-index/src/index_writer.rs"


def rule_codec_suffix_literal(files, problems, stats):
    for rel, raw in files:
        lines = blank_cfg_test(raw)
        for k, line in enumerate(lines):
            code = strip_comment(line)
            if not SUFFIX_LITERAL.search(code):
                continue
            stats["suffix_sites"] += 1
            if rel.replace(os.sep, "/") == SUFFIX_OWNER:
                continue
            problems.append(
                f"{rel}:{k + 1}: a `LuceneNN_N` per-field codec suffix literal "
                f"outside `index_writer::per_field_codec_suffix`. Derive it, "
                f"don't spell it. (docs/mechanical-gates.md#codec-suffix-literal)"
            )


# --------------------------------------------------------------------------
# Rule 4: the infallible blocktree lookups stay retired
# --------------------------------------------------------------------------
#
# c1 added `try_seek_exact`/`try_next`/`try_seek_ceil`/`try_current`, which
# surface a corrupt `.tim` block as an error instead of degrading it to "no
# such term". c39 finished the migration by hand -- marking the four infallible
# spellings `#[deprecated]` and rebuilding -- and `blocktree.rs`'s method docs
# describe the rule, but nothing ran it, so the migration could silently
# regress on the next batch that reached for the shorter name.

BLOCKTREE_RETIRED = ("seek_exact", "seek_ceil", "current")
BLOCKTREE_OWNER = "crates/lucene-codecs/src/blocktree.rs"
# Only files that actually consume the blocktree API. A file that never names
# `blocktree` cannot be calling `FieldTerms`/`TermsEnum`, and `fst.rs` has
# same-named methods of its own.
BLOCKTREE_IMPORT = re.compile(r"use\s+(lucene_codecs|crate)::blocktree|\bblocktree::")


def rule_blocktree_infallible(files, problems, stats):
    pat = re.compile(
        r"(?P<recv>[A-Za-z_][\w.]*)\.(?P<m>" + "|".join(BLOCKTREE_RETIRED) + r")\s*\("
    )
    for rel, raw in files:
        if rel.replace(os.sep, "/") == BLOCKTREE_OWNER:
            continue  # where both spellings are defined
        if not BLOCKTREE_IMPORT.search("\n".join(raw)):
            continue
        lines = blank_cfg_test(raw)
        for k, line in enumerate(lines):
            code = strip_comment(line)
            for m in pat.finditer(code):
                stats["blocktree_sites"] += 1
                problems.append(
                    f"{rel}:{k + 1}: `{m.group('recv')}.{m.group('m')}(..)` in a "
                    f"module that consumes `blocktree`. Use the `try_` form, which "
                    f"reports a corrupt `.tim` block instead of answering "
                    f"'no such term'. "
                    f"(docs/mechanical-gates.md#blocktree-infallible)"
                )


# --------------------------------------------------------------------------
# Rule 5: no *new* per-document re-derivation of a doc-values column
# --------------------------------------------------------------------------
#
# `doc_values::numeric_value`/`binary_value` re-derive a column's addressing
# from its `NumericEntry`/`BinaryEntry` on every call; `NumericReader`/
# `BinaryReader` derive it once and are the sanctioned multi-lookup API. The
# "call the free function once per document" defect has already shipped twice
# (b13's `soft_deletes::effective_live_docs`, c14's column merge).
#
# The existing sites are a burn-down list, not a clean bill of health, and they
# are keyed by (file, enclosing fn) rather than by line so ordinary edits do not
# churn it. A new one fails the gate; a migrated one fails it too, asking for
# the count to come down. This is the same shape as
# `docs/arithmetic-gate.md`'s `TODO(arith-audit)` markers, for the same reason:
# the debt has to be visible and it has to be able only to shrink.

DV_FREE_CALL = re.compile(r"doc_values::(numeric_value|binary_value)\s*\(")
LOOP_HEADER = re.compile(r"^\s*(for|while)\b")
ITERATOR_HEADER = re.compile(r"\.(iter|map|filter_map|for_each|flat_map)\s*\(")
DV_LOOP_BURNDOWN = {
    ("crates/lucene-index/src/check_index.rs", "check_doc_values"): 2,
    ("crates/lucene-index/src/check_index.rs", "doc_values_presence"): 1,
    ("crates/lucene-index/src/check_index.rs", "sort_key_values"): 1,
    ("crates/lucene-index/src/merge.rs", "merge_binary_doc_values"): 1,
    ("crates/lucene-search/src/doc_value_query.rs", "search_numeric_range"): 1,
    ("crates/lucene-search/src/doc_value_query.rs", "search_numeric_range_with_skip_index"): 1,
    ("crates/lucene-search/src/doc_value_query.rs", "sort_by_numeric_doc_value"): 1,
    ("crates/lucene-search/src/doc_value_query.rs", "sort_top_n_by_numeric_doc_value"): 1,
    ("crates/lucene-search/src/facets.rs", "count_single_valued"): 1,
}


def rule_doc_values_per_doc(files, problems, stats):
    found = {}
    for rel, raw in files:
        relp = rel.replace(os.sep, "/")
        lines = blank_cfg_test(raw)
        for fname, a, b in fn_spans(lines):
            for k in range(a, b + 1):
                if not DV_FREE_CALL.search(strip_comment(lines[k])):
                    continue
                # Walk the *ancestor chain*: every strictly-less-indented line
                # above the call, outwards to the `fn` header. A loop or
                # iterator-adaptor header anywhere on that chain means this
                # call runs once per document. Stopping at the first
                # less-indented line is not enough -- the call is routinely
                # nested inside a `present.push(..)` or a `match` arm.
                threshold = len(lines[k]) - len(lines[k].lstrip())
                for j in range(k - 1, a - 1, -1):
                    line = lines[j]
                    if not line.strip():
                        continue
                    indent = len(line) - len(line.lstrip())
                    if indent >= threshold:
                        continue
                    threshold = indent
                    if LOOP_HEADER.match(line) or ITERATOR_HEADER.search(line):
                        found[(relp, fname)] = found.get((relp, fname), 0) + 1
                        break
    stats["dv_loop_sites"] = sum(found.values())
    for key, n in sorted(found.items()):
        expected = DV_LOOP_BURNDOWN.get(key, 0)
        if n > expected:
            problems.append(
                f"{key[0]}: `fn {key[1]}` calls the free "
                f"`doc_values::numeric_value`/`binary_value` per document "
                f"{n} time(s), {expected} recorded. Use `NumericReader`/"
                f"`BinaryReader`, which derive the column's addressing once. "
                f"(docs/mechanical-gates.md#doc-values-per-doc)"
            )
    for key, expected in sorted(DV_LOOP_BURNDOWN.items()):
        n = found.get(key, 0)
        if n < expected:
            problems.append(
                f"{key[0]}: `fn {key[1]}` is on the doc-values burn-down list "
                f"with {expected} per-document call(s) but has {n}. Lower the "
                f"count in `DV_LOOP_BURNDOWN` (and in "
                f"docs/mechanical-gates.md) in the same change. "
                f"(docs/mechanical-gates.md#doc-values-per-doc)"
            )


# --------------------------------------------------------------------------
# Rule 6: the sweep ledger has exactly one list to plan from
# --------------------------------------------------------------------------
#
# `docs/sweep/m2/LEDGER.md` carries a reconciled "Open work, prioritised"
# section and, below it, a frozen archive of where each finding was first
# raised. Batches repeatedly closed the prioritised entry and left its archive
# twin sitting open -- which is the drift c34 was run to remove, and which
# still misled **six** later batches after c34 ran, because 16 duplicate boxes
# survived that reconciliation.
#
# The invariant is the smallest one that makes the failure impossible to
# express: **the file contains no `- [ ]`.** An archive entry is `- [x]`
# (closed, with the evidence), `- [~]` (obsolete) or `- [->]` (open, and
# tracked as a numbered open-work item, which it names). The prioritised list
# itself is numbered, not checkboxed, so it is unaffected.
#
# This is a rule about a document, not about code. It lives here because there
# is no other gate a document can fail, and because the defect it prevents has
# cost this sweep more batches than any single code defect has.

LEDGER = "docs/sweep/m2/LEDGER.md"
OPEN_BOX = re.compile(r"^\s*- \[ \]")


def rule_ledger_single_list(_files, problems, stats):
    path = os.path.join(ROOT, LEDGER)
    if not os.path.exists(path):
        problems.append(f"{LEDGER}: missing (this rule assumes the sweep ledger exists)")
        return
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if OPEN_BOX.match(line):
                stats["ledger_open_boxes"] += 1
                problems.append(
                    f"{LEDGER}:{lineno}: an unticked `- [ ]`. The archive below "
                    f"\"Open work, prioritised\" is not a to-do list: close it "
                    f"(`- [x]`, naming the batch and the evidence in the tree), "
                    f"mark it obsolete (`- [~]`), or point it at its numbered "
                    f"open-work item (`- [->]`). "
                    f"(docs/mechanical-gates.md#ledger-single-list)"
                )


RULES = (
    ("fixed-bitset-bound", rule_fixed_bitset_bound),
    ("sentinel-callers", rule_sentinel_callers),
    ("codec-suffix-literal", rule_codec_suffix_literal),
    ("blocktree-infallible", rule_blocktree_infallible),
    ("doc-values-per-doc", rule_doc_values_per_doc),
    ("ledger-single-list", rule_ledger_single_list),
)


def main(argv):
    verbose = "--verbose" in argv
    only = None
    for arg in argv:
        if arg.startswith("--only="):
            only = arg.split("=", 1)[1]
    files = src_files()
    problems = []
    stats = {
        "fbs_sites": 0,
        "fbs_proofs": 0,
        "sentinel_sites": 0,
        "sentinel_waived": 0,
        "suffix_sites": 0,
        "blocktree_sites": 0,
        "dv_loop_sites": 0,
        "ledger_open_boxes": 0,
    }
    for name, rule in RULES:
        if only and name != only:
            continue
        rule(files, problems, stats)

    if verbose:
        print(f"files scanned                       : {len(files)}")
        print(f"FixedBitSet index sites             : {stats['fbs_sites']}")
        print(f"  ... carrying an // FBS: proof     : {stats['fbs_proofs']}")
        print(f"sentinel call sites                 : {stats['sentinel_sites']}")
        print(f"  ... waived with // SENTINEL-OK:   : {stats['sentinel_waived']}")
        print(f"codec-suffix literals               : {stats['suffix_sites']}")
        print(f"blocktree infallible lookups        : {stats['blocktree_sites']}")
        print(f"doc-values per-document call sites  : {stats['dv_loop_sites']}" + f" (burn-down: {sum(DV_LOOP_BURNDOWN.values())})")

    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print(
            f"\ncheck-port-invariants: {len(problems)} violation(s); "
            f"see docs/mechanical-gates.md",
            file=sys.stderr,
        )
        return 1
    print("check-port-invariants: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
