import re, json, sys, glob, os
root = sys.argv[1]
out = []
lit = r'"(?:[^"\\\n]|\\.)*"'
# exec( "lit" (+ "lit")* ) or exec("lit" ..., more args)
pat = re.compile(r'\bexec\(\s*((?:' + lit + r'\s*\+\s*)*' + lit + r')\s*([,)])')
def unq(s):
    s = s[1:-1]
    return bytes(s, 'utf-8').decode('unicode_escape') if '\\u' not in s else json.loads('"' + s.replace('\\\'', "'") + '"')
for f in sorted(glob.glob(os.path.join(root, '**/*.java'), recursive=True)):
    src = open(f).read()
    for m in pat.finditer(src):
        parts = re.findall(lit, m.group(1))
        try:
            text = ''.join(json.loads(p.replace("\\'", "'")) for p in parts)
        except Exception:
            continue
        out.append({'file': os.path.basename(f), 'src': text, 'more_args': m.group(2) == ','})
seen = set(); res = []
for o in out:
    if o['src'] in seen: continue
    seen.add(o['src']); res.append(o)
for o in res:
    print(json.dumps(o))
