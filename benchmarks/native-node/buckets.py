"""Bucket async-profiler collapsed stacks (with -t thread frames) by OpenSearch request stage.

  buckets.py FILE.collapsed [--top BUCKET N]

A stack goes to the first rule (in order) any of whose patterns occurs in it, so the
specific stages (native engine, fetch, reduce, response rendering) come before the
generic ones (netty, coordinator action).
"""
import collections, sys

RULES = [
    ("gc/jit/vm", ["Compile::", "PhaseChaitin", "PhaseIdealLoop", "PhaseCFG", "PhaseIFG", "Matcher::", "ciEnv", "CompileBroker", "C2Compiler", "C1 Runtime", "Parse::", "[GC Thread", "[G1 ", "[C2 Compiler", "[C1 Compiler", "[VM Thread", "[VM Periodic", "G1CollectedHeap", "[Service Thread"]),
    ("fetch: _source filter/parse", ["FetchSourcePhase", "SourceLookup", "XContentMapValues", "filterSource", "SourceFieldMapper"]),
    ("fetch: stored fields / docs", ["FetchPhase", "StoredFieldsReader", "StoredFieldVisitor", "FieldsVisitor", "LZ4", "Lucene90CompressingStoredFields"]),
    ("plugin Java glue (encode/decode/replay)", ["org/lucenerust"]),
    ("query phase (Lucene/Java)", ["QueryPhase", "IndexSearcher.search", "ContextIndexSearcher", "BulkScorer", "Scorer", "TopScoreDocCollector",
                                  "KNNQuery", "KnnFloatVectorQuery", "AbstractKnnVectorQuery", "HnswGraphSearcher"]),
    ("shard: context/parse/rewrite", ["SearchService.createContext", "SearchService.createAndPutReaderContext", "DefaultSearchContext", "SearchService.parseSource",
                                     "QueryShardContext", "QueryBuilder.toQuery", "Rewriteable", "IndexShard.acquireSearcher", "SearchService.executeQueryPhase",
                                     "SearchService.executeFetchPhase", "SearchService.loadOrExecuteQueryPhase", "SearchService"]),
    ("coord: reduce/merge", ["SearchPhaseController", "QueryPhaseResultConsumer", "TopDocs.merge"]),
    ("response: toXContent/JSON", ["toXContent", "XContentBuilder", "jackson", "RestToXContentListener", "RestResponse", "BytesRestResponse"]),
    ("rest: request parse", ["RestSearchAction", "SearchSourceBuilder.parseXContent", "parseInnerQueryBuilder", "XContentParser", "RestController", "RestHandler"]),
    ("coord: search action", ["TransportSearchAction", "AbstractSearchAsyncAction", "SearchQueryThenFetchAsyncAction", "FetchSearchPhase", "ExpandSearchPhase",
                              "TransportService", "SearchTransportService", "TaskManager"]),
    ("http/netty io", ["io.netty", "Netty4", "epoll", "sun.nio"]),
    ("threadpool/park", ["ThreadPoolExecutor", "LinkedTransferQueue", "park", "futex"]),
]

JVM_NATIVE = ("Unsafe", "JVM_", "SharedRuntime", "OptoRuntime", "pthread", "clock_gettime", "vdso", "Interpreter", "stub", "jni_", "JNI", "os::", "G1", "ObjAlloc", "MemAllocator", "InstanceKlass", "Runtime1")

def native_engine(frames):
    """The leaf frames below the deepest Java frame are Rust (not JVM runtime), under plugin code."""
    j = max((i for i, f in enumerate(frames) if "/" in f and not f.startswith("[")), default=-1)
    if j < 0 or j == len(frames) - 1:
        return False
    tail = frames[j + 1:]
    if any(any(k in f for k in JVM_NATIVE) for f in tail):
        return False
    return any("lucenerust" in f for f in frames[:j + 1]) or any("::" in f for f in tail)

def bucket(stack):
    frames = stack.split(";")
    if "[GC Thread" not in stack and "Compiler" not in frames[0] and native_engine(frames):
        return "native engine (Rust)"
    luc = [f for f in frames if "lucenerust" in f]
    filter_only = bool(luc) and luc[-1].endswith("Filter.apply")
    for name, pats in RULES:
        if filter_only and name.startswith("plugin Java glue"):
            continue
        for p in pats:
            if p in stack:
                return name
    return "other"

def main():
    f = sys.argv[1]
    tot = 0; B = collections.Counter(); ex = collections.defaultdict(collections.Counter)
    for line in open(f, errors="replace"):
        line = line.rstrip()
        if not line: continue
        stack, _, n = line.rpartition(" ")
        n = int(n); tot += n
        b = bucket(stack); B[b] += n
        leafs = stack.split(";")
        ex[b][";".join(leafs[-3:])] += n
    for b, n in B.most_common():
        print(f"{100*n/tot:5.1f}%  {b}")
    if "--top" in sys.argv:
        k = sys.argv[sys.argv.index("--top") + 1]; m = int(sys.argv[sys.argv.index("--top") + 2])
        for s, n in ex[k].most_common(m):
            print(f"   {100*n/tot:5.2f}% {s[-220:]}")

if __name__ == "__main__":
    main()
