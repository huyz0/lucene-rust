#!/usr/bin/env python3
"""`_id` sorts against a stock node: ids of every encoding (numeric, UTF-8, base64 like generated ones),
both orders, a filter, and a `search_after` walk. Hit ids and sort values must be identical, and the
plugin must answer natively (no new fallbacks).

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 python3 opensearch-plugin/e2e/verify_id_sort.py
"""
import os
import json,random,sys,urllib.request
def req(u,m,p,b=None,raw=False):
    d=None if b is None else (b if raw else json.dumps(b).encode())
    r=urllib.request.Request(u+p,data=d,method=m,headers={"content-type":"application/json"})
    try: return json.load(urllib.request.urlopen(r))
    except urllib.error.HTTPError as e: return {"err":e.read().decode()[:200]}
P,S=os.environ.get("PLUGIN_URL","http://localhost:9200"),os.environ.get("STOCK_URL","http://localhost:9201")
r=random.Random(1)
ids=[]
for i in range(60000):
    k=r.random()
    ids.append(__import__("base64").urlsafe_b64encode(bytes(r.getrandbits(8) for _ in range(15))).decode() if k<0.3 else ("%x"%r.getrandbits(r.choice([12,40,70])) if k<0.8 else "é%d-%s"%(i,"x"*r.randint(0,5))))
lines=[]
for i,x in enumerate(ids):
    lines.append(json.dumps({"index":({"_id":x} if x else {})})); lines.append(json.dumps({"n":i}))
body=("\n".join(lines)+"\n").encode()
for u in (P,S):
    req(u,"DELETE","/idx")
    req(u,"PUT","/idx",{"settings":{"number_of_shards":1,"number_of_replicas":0}})
    req(u,"POST","/idx/_bulk",body,True)
    req(u,"POST","/idx/_refresh")
    req(u,"POST","/idx/_forcemerge?max_num_segments=2")
bad=0;tot=0
for order in ("asc","desc"):
  for q in ({"match_all":{}},{"range":{"n":{"lt":30000}}}):
    for node_after in (False,True):
        res=[]
        for u in (P,S):
            seen=[];after=None
            for _ in range(4):
                b={"size":500,"query":q,"sort":[{"_id":order}],"_source":False}
                if after: b["search_after"]=after
                h=req(u,"POST","/idx/_search",b)
                if "err" in h: seen.append(h["err"]);break
                hs=h["hits"]["hits"]
                if not hs:break
                seen+=[(x["_id"],x["sort"]) for x in hs]
                after=hs[-1]["sort"]
                if not node_after: break
            res.append(seen)
        tot+=1
        if res[0]!=res[1]:
            bad+=1;k=next((j for j,(a,b) in enumerate(zip(*res)) if a!=b),None);print("MISMATCH",order,node_after,len(res[0]),len(res[1]),k,res[0][k:k+2] if k is not None else "",res[1][k:k+2] if k is not None else "")
print("id sorts",tot,"mismatches",bad)

sys.exit(1 if bad else 0)
