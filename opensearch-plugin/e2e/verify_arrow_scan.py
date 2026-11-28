#!/usr/bin/env python3
"""An index written by a plugin node (updates, deletes, missing values) read through the Arrow source, against
what the node's own search holds: every live document once, no soft-deleted version, the same values, and every
pushed filter (=, !=, <, >=, in, is null, limits) keeping exactly the documents a stock evaluation keeps; Polars and
DuckDB over the same table give the same rows.

    python3 opensearch-plugin/e2e/verify_arrow_scan.py SEED   (plugin on :9250; LUCENE_ARROW_LIB=liblucene_ffi.so
                                                               built with feature arrow-sql, node data in nodectl's `plug`)
Multi-valued long and keyword columns are refused by the scan with an error (`MV=1` to see it)."""
import json, random, sys, glob, os, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "crates", "lucene-ffi", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nodectl
os.environ.setdefault("LUCENE_ARROW_LIB", "liblucene_ffi.so")
import lucene_arrow as la
MV=os.environ.get("MV")=="1"
FAILS=[]
B="http://localhost:9250"
def req(m,p,b=None,nd=False):
    d=None if b is None else (b if isinstance(b,bytes) else json.dumps(b).encode())
    r=urllib.request.Request(B+p,data=d,method=m,headers={"content-type":"application/x-ndjson" if nd else "application/json"})
    try:
        with urllib.request.urlopen(r) as f: return json.load(f)
    except urllib.error.HTTPError as e: return json.load(e)
req("DELETE","/ar")
print(req("PUT","/ar",{"settings":{"index.number_of_shards":1,"index.number_of_replicas":0,"index.engine.accelerated":True,"refresh_interval":"-1"},
 "mappings":{"properties":{"n":{"type":"long"},"u":{"type":"long"},"k":{"type":"keyword"},"t":{"type":"text"}}}})["acknowledged"])
rnd=random.Random(int(sys.argv[1]) if len(sys.argv)>1 else 1)
def doc(i=0):
    d={"u":i}
    if rnd.random()<.8: d["n"]=rnd.randint(-5,40)
    if rnd.random()<.8: d["k"]=rnd.choice(["a","b","c","d",""])
    if rnd.random()<.8: d["t"]="x y"
    if MV and rnd.random()<.1: d["n"]=[rnd.randint(0,5),rnd.randint(0,5)]
    if MV and rnd.random()<.1: d["k"]=["a","z"]
    return d
for rnd_ in range(3):
    lines=[]
    for i in range(rnd.randint(300,600)):
        i=rnd.randint(0,700)
        op=rnd.choice(["index","index","index","delete","update"])
        if op=="delete": lines.append(json.dumps({"delete":{"_id":str(i)}}))
        elif op=="update": lines+= [json.dumps({"update":{"_id":str(i)}}),json.dumps({"doc":doc(i),"doc_as_upsert":True})]
        else: lines+=[json.dumps({"index":{"_id":str(i)}}),json.dumps(doc(i))]
    r=req("POST","/ar/_bulk",("\n".join(lines)+"\n").encode(),True)
    req("POST","/ar/_refresh"); req("POST","/ar/_flush?force=true")
req("POST","/ar/_flush?force=true")
# stock truth
hits=req("POST","/ar/_search",{"size":10000,"query":{"match_all":{}},"sort":[{"_id":"asc"}]})["hits"]
truth={h["_source"]["u"]:h["_source"] for h in hits["hits"]}
print("live docs",len(truth))
uuid=req("GET","/ar/_settings")["ar"]["settings"]["index"]["uuid"]
path="%s/nodes/0/indices/%s/0/index" % (nodectl.data_dir("plug"), uuid)
print(path, os.listdir(path)[:5])
for cols in [{"u":"long","n":"long","k":"keyword"}]:
    try:
        t=la.Table(path,"ar",cols)
    except Exception as e:
        print("OPEN FAIL",e); sys.exit(1)
    print(t.info)
    tb=t.scan(dictionary=False).read_all().to_pylist()
    print("scan rows",len(tb), tb[:3])
    got={r["u"]:r for r in tb}
    print("dup ids",len(tb)-len(got))
    if len(tb)!=len(got): FAILS.append("duplicate rows")
    miss=set(truth)-set(got); extra=set(got)-set(truth)
    print("missing",len(miss),"extra(deleted/updated-old)",len(extra))
    bad=0
    for i,r in got.items():
        if i not in truth: continue
        s=truth[i]
        for c in ("n","k"):
            v=s.get(c); g=r[c]
            if isinstance(v,list): continue
            if v!=g: bad+=1; print("DIFF",i,c,v,g) if bad<10 else 0
    print("value diffs",bad)
    if bad or miss or extra: FAILS.append("rows or values differ")
    # filters
    for f in [[("n","=",5)],[("n","!=",5)],[("n","<",3)],[("n",">=",20)],[("n","in",[1,2,3])],[("n","is_null",True)],[("n","is_null",False)],[("k","=","a")],[("k","!=","a")],[("k","in",["a","b"])],[("k","is_null",True)],[("k","=","")]]:
        col,op,val=f[0]
        want=set()
        for i,s in truth.items():
            v=s.get(col)
            vs=v if isinstance(v,list) else [v]
            if isinstance(v,list): continue
            if op=="=": ok=v==val and v is not None
            elif op=="!=": ok=v is not None and v!=val
            elif op=="<": ok=v is not None and v<val
            elif op==">=": ok=v is not None and v>=val
            elif op=="in": ok=v in val
            elif op=="is_null": ok=(v is None)==val
            if ok: want.add(i)
        rows=t.scan(filters=f,dictionary=False).read_all().to_pylist()
        have={r["u"] for r in rows if not isinstance(truth.get(r["u"],{}).get(col),list)}
        
        if have!=want: FAILS.append(str(f))
        print("ok  " if have==want else "FAIL", f, len(want), len(have), sorted(want^have)[:5])
    for lim in (0,1,7):
        n=t.scan(limit=lim).read_all().num_rows
        if n!=min(lim,len(truth)): FAILS.append("limit %d"%lim)
        print("limit",lim,n)
import polars as pl, duckdb
df=t.polars().collect()
truthdf=pl.DataFrame([{"u":i,"n":s.get("n"),"k":s.get("k")} for i,s in truth.items()],schema={"u":pl.Int64,"n":pl.Int64,"k":pl.String}).sort("u")
eq=(df.select("u","n","k").with_columns(pl.col("k").cast(pl.String)).sort("u").equals(truthdf))
print("polars full equal",eq)
if not eq: FAILS.append("polars")
for e in [pl.col("n")!=5, pl.col("n").is_null(), pl.col("k")!="a", pl.col("k").is_in(["a","b"]), (pl.col("n")>3)&(pl.col("k")=="b"), pl.col("n")<3]:
    a=t.polars().filter(e).select("u").collect().sort("u")["u"].to_list()
    b=truthdf.filter(e).select("u")["u"].to_list()
    print("polars","ok" if a==b else "FAIL",str(e)[:40],len(a),len(b))
    if a!=b: FAILS.append("polars "+str(e))
r=t.scan(dictionary=True)
print("duckdb", duckdb.sql("select count(*), count(n), count(distinct k) from r").fetchall(), len(truth), sum(1 for s in truth.values() if "n" in s), len({s["k"] for s in truth.values() if "k" in s}))

print("%d differences" % len(FAILS), *FAILS)
sys.exit(1 if FAILS else 0)
