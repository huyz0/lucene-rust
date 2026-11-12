"""Dumps every dashboard's aggregations at several times, the terms exact (a shard_size past
every key) and as asked, for comparing engines over the same documents."""
import json,sys,os
OUTP=sys.argv[1]
sys.argv=["x","http://localhost:9250","corpus","parity","/dev/null"]
os.environ["SKIP_STATIC"]="1"
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")).read(); src=src[:src.index("# Live: each round")]
exec(compile(src,"bench","exec"))
def exact(a):
    if isinstance(a,dict):
        a={k:exact(v) for k,v in a.items()}
        if "terms" in a and isinstance(a["terms"],dict) and "field" in a["terms"]:
            a["terms"]["shard_size"]=200000
        return a
    return a
out={}
for e in (end, end-5*H-123456, end-50*H-777, end-100*H-9):
    for name,b in dashboards(e).items():
        out[f"{e}/{name}/asked"]=req("POST","/logs/_search?request_cache=false",b)["aggregations"]
        b2=dict(b); b2["aggs"]=exact(b["aggs"])
        out[f"{e}/{name}/exact"]=req("POST","/logs/_search?request_cache=false",b2)["aggregations"]
json.dump(out,open(OUTP,"w"),sort_keys=True)
