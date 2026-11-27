#!/usr/bin/env python3
"""Generate lucene-painless's allowlist table from OpenSearch's Painless allowlist files.

`lang-painless` types a script against `spi/*.txt` (what each class, method,
constructor and field a script may name) and, by Java reflection, the
hierarchy of those classes (a class gets the members of every allowlisted
class it is assignable to). The Rust front end types against the same
facts; this script reads them once into `front/allowlist.tsv` so the Rust side
never reads Java and a change of allowlist is a reviewable diff.

  gen-painless-allowlist.py SPI_DIR            write the table
  gen-painless-allowlist.py --check SPI_DIR    fail when the committed table is stale

SPI_DIR is `modules/lang-painless/src/main/resources/org/opensearch/painless/spi`
of an OpenSearch checkout (or the same files unpacked from the module's jar).
The hierarchy is read from the JDK on the PATH for the `java.*` classes; the
`org.opensearch.*` classes get theirs when the contexts that own them are
served.

Rows (tab separated; types are canonical: a binary class name, a primitive,
`def` or `void`, with `[]` for each array dimension; a parameter list is the
types joined by `,`):

  class   NAME  FILE  FLAGS                      FLAGS: `no_import` or `-`
  parent  CHILD PARENT                           every allowlisted class CHILD is assignable to
  ctor    CLASS PARAMS  FILE
  method  CLASS NAME RET PARAMS FLAGS FILE       FLAGS: `static`, `nondeterministic`,
                                                 `augment=CLASS` (comma joined) or `-`
  field   CLASS NAME TYPE FLAGS FILE             FLAGS: `static`, `final` or `-`
  iface   CLASS NAME RET PARAMS                  the one abstract method of a functional interface
  import  NAME  RET PARAMS KIND TARGET FILE      static_import: KIND `from_class` or `bound_to`
"""
import os
import re
import subprocess
import sys
import tempfile

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "crates",
    "lucene-painless",
    "src",
    "front",
    "allowlist.tsv",
)
PRIMS = {"boolean", "byte", "short", "char", "int", "long", "float", "double", "def", "void"}

REFLECT_JAVA = """
import java.io.*;
import java.lang.reflect.*;
import java.util.*;
public class Reflect {
  static Class<?> cls(String t) throws Exception {
    int dims = 0;
    while (t.endsWith("[]")) { t = t.substring(0, t.length() - 2); dims++; }
    Class<?> c;
    switch (t) {
      case "boolean": c = boolean.class; break; case "byte": c = byte.class; break;
      case "short": c = short.class; break; case "char": c = char.class; break;
      case "int": c = int.class; break; case "long": c = long.class; break;
      case "float": c = float.class; break; case "double": c = double.class; break;
      case "def": c = Object.class; break;
      default: c = Class.forName(t, false, Reflect.class.getClassLoader());
    }
    for (int i = 0; i < dims; i++) c = Array.newInstance(c, 0).getClass();
    return c;
  }
  static String tn(Class<?> c) {
    if (c.isArray()) return tn(c.getComponentType()) + "[]";
    return c.getName();
  }
  public static void main(String[] a) throws Exception {
    BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
    List<String> classes = new ArrayList<>();
    List<String[]> reqs = new ArrayList<>();
    for (String l; (l = r.readLine()) != null; ) {
      String[] f = l.split("\\t", -1);
      if (f[0].equals("C")) classes.add(f[1]); else reqs.add(f);
    }
    Map<String, Class<?>> cs = new LinkedHashMap<>();
    for (String n : classes) {
      try { cs.put(n, cls(n)); } catch (Throwable t) { System.out.println("missing\t" + n); }
    }
    for (String c : cs.keySet()) for (String p : cs.keySet())
      if (!c.equals(p) && cs.get(p).isAssignableFrom(cs.get(c))) System.out.println("parent\t" + c + "\t" + p);
    for (String n : cs.keySet()) {
      Class<?> c = cs.get(n);
      if (!c.isInterface()) continue;
      Method only = null;
      int count = 0;
      for (Method m : c.getMethods()) {
        if (!Modifier.isAbstract(m.getModifiers())) continue;
        boolean object = false;
        try { Object.class.getMethod(m.getName(), m.getParameterTypes()); object = true; } catch (NoSuchMethodException e) { }
        if (object) continue;
        boolean dup = false;
        if (only != null && only.getName().equals(m.getName()) && Arrays.equals(only.getParameterTypes(), m.getParameterTypes())) dup = true;
        if (!dup) { only = m; count++; }
      }
      if (count == 1) {
        StringBuilder ps = new StringBuilder();
        for (Class<?> p : only.getParameterTypes()) { if (ps.length() > 0) ps.append(','); ps.append(tn(p)); }
        System.out.println("iface\t" + n + "\t" + only.getName() + "\t" + tn(only.getReturnType()) + "\t" + ps);
      }
    }
    for (String[] q : reqs) {
      Class<?> c = cs.get(q[1]);
      String res = "?";
      if (c != null) {
        try {
          if (q[0].equals("M")) {
            List<Class<?>> ps = new ArrayList<>();
            if (!q[3].isEmpty()) for (String p : q[3].split(",")) ps.add(cls(p));
            res = Modifier.isStatic(c.getMethod(q[2], ps.toArray(new Class<?>[0])).getModifiers()) ? "1" : "0";
          } else if (q[0].equals("F")) {
            res = Modifier.isStatic(c.getField(q[2]).getModifiers()) ? "1" : "0";
          }
        } catch (NoSuchMethodException | NoSuchFieldException e) { res = "none"; }
      }
      System.out.println("static\t" + String.join("\t", q) + "\t" + res);
    }
  }
}
"""


def strip(line):
    i = line.find("#")
    return (line if i < 0 else line[:i]).strip()


def read_files(spi):
    files = sorted(f for f in os.listdir(spi) if f.endswith(".txt"))
    return [(f[: -len(".txt")], open(os.path.join(spi, f), encoding="utf-8").read().splitlines()) for f in files]


def canonical_short(full):
    return full.rsplit(".", 1)[-1].replace("$", ".")


def parse(files):
    classes = {}  # full -> (file, no_import)
    for name, lines in files:
        for line in lines:
            m = re.match(r"^class\s+(\S+)(\s+@no_import)?\s*\{", line)
            if m:
                classes[m.group(1)] = (name, bool(m.group(2)))
    shorts = {}
    for full, (_, no_import) in classes.items():
        if not no_import:
            shorts[canonical_short(full)] = full
    for p in PRIMS:
        shorts[p] = p

    def ty(t):
        t = t.strip()
        dims = 0
        while t.endswith("[]"):
            t = t[:-2]
            dims += 1
        if t in shorts:
            base = shorts[t]
        elif t in classes:
            base = t
        else:
            sys.exit(f"gen-painless-allowlist: unknown type [{t}]")
        return base + "[]" * dims

    def params(s):
        s = s.strip()
        return ",".join(ty(p) for p in s.split(",")) if s else ""

    rows = {"class": [], "ctor": [], "method": [], "field": [], "import": []}
    for name, lines in files:
        cur = None
        in_imports = False
        for raw in lines:
            line = strip(raw)
            if not line:
                continue
            m = re.match(r"^class\s+(\S+)(\s+@no_import)?\s*\{", line)
            if m:
                cur = m.group(1)
                rows["class"].append((cur, name, "no_import" if m.group(2) else "-"))
                continue
            if line.startswith("static_import"):
                in_imports = True
                continue
            if line == "}":
                cur = None
                in_imports = False
                continue
            flags = []
            if "@nondeterministic" in line:
                flags.append("nondeterministic")
            line = re.sub(r"@\w+(\[[^\]]*\])?", "", line).strip()
            if in_imports:
                m = re.match(r"^(\S+)\s+(\w+)\(([^)]*)\)\s*(from_class|bound_to)\s+(\S+)$", line)
                if not m:
                    sys.exit(f"gen-painless-allowlist: {name}: cannot read import [{raw.strip()}]")
                rows["import"].append((m.group(2), ty(m.group(1)), params(m.group(3)), m.group(4), m.group(5), name))
                continue
            if cur is None:
                sys.exit(f"gen-painless-allowlist: {name}: member outside a class [{raw.strip()}]")
            if "(" in line:
                head, rest = line.split("(", 1)
                ps = rest.rsplit(")", 1)[0]
                parts = head.split()
                static = parts and parts[0] == "static"
                if static:
                    parts = parts[1:]
                if len(parts) <= 1:
                    rows["ctor"].append((cur, params(ps), name))
                    continue
                aug = None
                if len(parts) == 2:
                    ret, mname = parts
                elif len(parts) == 3:
                    ret, aug, mname = parts
                else:
                    sys.exit(f"gen-painless-allowlist: {name}: cannot read method [{raw.strip()}]")
                if static:
                    flags.insert(0, "static")
                if aug:
                    flags.append("augment=" + aug)
                rows["method"].append((cur, mname, ty(ret), params(ps), ",".join(flags) or "-", name))
            else:
                parts = line.split()
                fl = [p for p in parts[:-2] if p in ("static", "final")]
                rows["field"].append((cur, parts[-1], ty(parts[-2]), ",".join(fl) or "-", name))
    return classes, rows


# Classes the JDK cannot load here (OpenSearch's own): which of their members
# are static, as read from their Java sources. Every other member of these
# classes is an instance member.
STATIC_OVERRIDES = {
    ("org.opensearch.painless.api.Debug", "explain"),
    ("org.opensearch.search.aggregations.pipeline.MovingFunctions", "*"),
}


def reflect(classes, rows):
    names = [c for c in classes if c not in PRIMS]
    lines = ["C\t" + n for n in names]
    for c, name, _ret, params, flags, _f in rows["method"]:
        if "augment=" not in flags:
            lines.append("M\t%s\t%s\t%s" % (c, name, params))
    for c, name, _t, _fl, _f in rows["field"]:
        lines.append("F\t%s\t%s" % (c, name))
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "Reflect.java")
        open(src, "w").write(REFLECT_JAVA)
        out = subprocess.run(
            ["java", src],
            input="\n".join(lines) + "\n",
            capture_output=True,
            text=True,
            check=True,
            cwd=d,
        ).stdout
    parents, missing, static, ifaces = [], [], {}, []
    for line in out.splitlines():
        f = line.split("\t")
        if f[0] == "parent":
            parents.append((f[1], f[2]))
        elif f[0] == "iface":
            ifaces.append((f[1], f[2], f[3], f[4]))
        elif f[0] == "missing":
            missing.append(f[1])
        elif f[0] == "static":
            # static, M|F, class, name, [params,] result
            static[(f[1], f[2], f[3], f[4] if f[1] == "M" else "")] = f[-1]
    return sorted(parents), sorted(missing), static, sorted(ifaces)


def is_static(kind, cls, name, params, static):
    res = static.get((kind, cls, name, params if kind == "M" else ""), "?")
    if res == "none":
        sys.exit(f"gen-painless-allowlist: {cls}.{name}({params}) is not in the JDK class")
    if res == "?":
        return (cls, name) in STATIC_OVERRIDES or (cls, "*") in STATIC_OVERRIDES
    return res == "1"


def render(spi):
    files = read_files(spi)
    classes, rows = parse(files)
    parents, missing, static, ifaces = reflect(classes, rows)
    methods = []
    for c, name, ret, params, flags, f in rows["method"]:
        fl = [x for x in flags.split(",") if x != "-"]
        if "augment=" not in flags and is_static("M", c, name, params, static):
            fl.insert(0, "static")
        methods.append((c, name, ret, params, ",".join(fl) or "-", f))
    fields = []
    for c, name, t, _fl, f in rows["field"]:
        fields.append((c, name, t, "static" if is_static("F", c, name, "", static) else "-", f))
    out = [
        "# Generated by scripts/gen-painless-allowlist.py from Painless's spi/*.txt. Do not edit.",
        "# Hierarchy and static-ness are read from the JDK. Not loadable there: %d OpenSearch classes"
        " (their members are instance members, but Debug.explain and MovingFunctions: the source's)." % len(missing),
    ]
    out += ["class\t%s\t%s\t%s" % r for r in rows["class"]]
    out += ["parent\t%s\t%s" % r for r in parents]
    out += ["iface\t%s\t%s\t%s\t%s" % r for r in ifaces]
    out += ["ctor\t%s\t%s\t%s" % r for r in rows["ctor"]]
    out += ["method\t%s\t%s\t%s\t%s\t%s\t%s" % r for r in methods]
    out += ["field\t%s\t%s\t%s\t%s\t%s" % r for r in fields]
    out += ["import\t%s\t%s\t%s\t%s\t%s\t%s" % r for r in rows["import"]]
    return "\n".join(out) + "\n"


def main(argv):
    check = "--check" in argv
    args = [a for a in argv if a != "--check"]
    if len(args) != 1:
        sys.exit(__doc__)
    text = render(args[0])
    if check:
        have = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if have != text:
            sys.exit("gen-painless-allowlist: front/allowlist.tsv is stale; rerun without --check")
        print("gen-painless-allowlist: ok")
    else:
        with open(OUT, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(OUT)}: {text.count(chr(10))} lines")


if __name__ == "__main__":
    main(sys.argv[1:])
