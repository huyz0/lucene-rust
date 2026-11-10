import sys, time, numpy as np, hnswlib
pre=sys.argv[1]; efs=[int(x) for x in sys.argv[2].split(',')]; nq=int(sys.argv[3])
def rd(p,dt):
    b=np.fromfile(p,dtype=np.uint8); n,d=np.frombuffer(b[:8],dtype=np.uint32); return np.frombuffer(b[8:],dtype=dt).reshape(n,d)
base=rd(pre+'.base.fbin',np.float32); q=rd(pre+'.query.fbin',np.float32); gt=rd(pre+'.gt.ibin',np.int32)
idx=hnswlib.Index(space='cosine',dim=base.shape[1]); idx.init_index(max_elements=len(base),M=16,ef_construction=100,random_seed=42)
idx.set_num_threads(1); t=time.time(); idx.add_items(base,np.arange(len(base)),num_threads=1); print('build_s',round(time.time()-t,1),flush=True)
print('ef,recall@10,mean_us')
for ef in efs:
    idx.set_ef(max(ef,10)); idx.knn_query(q[nq:nq+200],k=10,num_threads=1)
    lat=[]; found=0
    for i in range(nq):
        t=time.perf_counter(); l,_=idx.knn_query(q[i:i+1],k=10,num_threads=1); lat.append(time.perf_counter()-t)
        found+=len(set(l[0]).intersection(gt[i,:10]))
    # batch timing minus python overhead
    t=time.perf_counter(); idx.knn_query(q[:nq],k=10,num_threads=1); bt=(time.perf_counter()-t)/nq
    print(f"{ef},{found/(nq*10):.4f},{np.mean(lat)*1e6:.0f},batch_us={bt*1e6:.0f}",flush=True)
