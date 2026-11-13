"""Compares parity.py dumps (run from the results directory): stock, main, cache-cold, cache."""
import json
def cmp(x,y,path,out):
    if isinstance(x,dict) and isinstance(y,dict):
        for k in set(x)|set(y): cmp(x.get(k),y.get(k),path+"/"+k,out)
    elif isinstance(x,list) and isinstance(y,list) and len(x)==len(y):
        for i,(p,q) in enumerate(zip(x,y)): cmp(p,q,f"{path}[{i}]",out)
    elif isinstance(x,float) or isinstance(y,float):
        if x is None or y is None or abs(x-y)>1e-9*max(1,abs(x)): out.append((path,x,y))
    elif x!=y: out.append((path,x,y))
d={k:json.load(open(f"parity-{k}.json")) for k in ("stock","main","cache-cold","cache")}
for a,b in (("stock","cache"),("stock","main"),("main","cache"),("cache-cold","cache")):
    for kind in ("exact","asked"):
        keys=[k for k in d[a] if k.endswith(kind)]
        bad={}; bitwise=sum(d[a][k]==d[b][k] for k in keys)
        for k in keys:
            o=[];cmp(d[a][k],d[b][k],"",o)
            if o: bad[k]=o
        print(a,b,kind,len(keys),"bitwise-equal",bitwise,"differ",len(bad), [ (k,v[:2]) for k,v in list(bad.items())[:1]])
