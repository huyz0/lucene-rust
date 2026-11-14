# Grouped k-NN (`group_by`, `group_size`)

Run `cargo run --release -p lucene-search --example knn_grouped -- PREFIX
--groups FILE --format hnsw|ivf --ef 10,20,50,100 --size 1,3 --over 3,10,30,100`.
The example indexes the vectors once through `IndexWriter` (four segments,
the group as the numeric doc value `g`) and measures, per query, the best
`size` documents of each of the best 10 groups three ways:

- `collapse_top_k`: a top-10 search collapsed by group, what stock k-NN with
  `collapse` returns;
- `oversample_then_group`: a top `10 * size * over` search grouped afterwards,
  the client-side workaround (stock `knn` with a large `k` plus `collapse`
  inner hits);
- `group_by`: `knn_top_hits_grouped`, the plugin's `group_by` / `group_size`.

Recall is against the exhaustive answer; `groups_found` is how many of the 10
groups the hits cover. Means over 200 queries on one core of a 4-vCPU cloud
container, after 30 warm-up queries.

## Data

GloVe-100 (Hugging Face `glove100.npy`), normalized, 390,000 base rows and
1,000 held-out queries, dot product.

- `topics`: each row's nearest of 512 sampled rows; 512 large groups of alike
  vectors (up to 7,254 rows), spread over every segment.
- `chunks`: 86,573 groups of 1 to 8 rows from the same of 4,096 clusters,
  as a document's chunks are; mostly within one segment.

## Results (selected)

| groups | format | size | method | setting | groups | recall | mean µs |
|---|---|---:|---|---|---:|---:|---:|
| topics | hnsw | 1 | collapse_top_k | ef=100 | 5.4 | 0.521 | 666 |
| topics | hnsw | 1 | oversample_then_group | k=300 | 9.5 | 0.906 | 1,273 |
| topics | hnsw | 1 | oversample_then_group | k=1000 | 9.9 | 0.973 | 2,936 |
| topics | hnsw | 1 | group_by | ef=20 | 10 | 0.944 | 1,705 |
| topics | hnsw | 1 | group_by | ef=50 | 10 | 0.979 | 3,051 |
| topics | hnsw | 3 | oversample_then_group | k=900 | 9.8 | 0.909 | 2,474 |
| topics | hnsw | 3 | oversample_then_group | k=3000 | 10 | 0.972 | 6,081 |
| topics | hnsw | 3 | group_by | ef=10 | 10 | 0.937 | 2,120 |
| topics | hnsw | 3 | group_by | ef=50 | 10 | 0.983 | 4,979 |
| chunks | hnsw | 1 | oversample_then_group | k=30 | 10 | 0.899 | 354 |
| chunks | hnsw | 1 | oversample_then_group | k=100 | 10 | 0.945 | 693 |
| chunks | hnsw | 1 | group_by | ef=50 | 10 | 0.922 | 798 |
| chunks | hnsw | 1 | group_by | ef=100 | 10 | 0.949 | 1,344 |
| chunks | hnsw | 3 | oversample_then_group | k=3000 | 10 | 0.736 | 6,284 |
| chunks | hnsw | 3 | group_by | ef=10 | 10 | 0.858 | 432 |
| chunks | hnsw | 3 | group_by | ef=100 | 10 | 0.955 | 1,368 |
| topics | ivf | 1 | oversample_then_group | k=1000 | 9.9 | 0.972 | 6,880 |
| topics | ivf | 1 | group_by | ef=10 | 10 | 0.984 | 6,404 |
| topics | ivf | 3 | oversample_then_group | k=3000 | 10 | 0.954 | 10,474 |
| topics | ivf | 3 | group_by | ef=10 | 10 | 0.986 | 9,012 |
| chunks | ivf | 1 | oversample_then_group | k=30 | 10 | 0.989 | 1,361 |
| chunks | ivf | 1 | group_by | ef=10 | 10 | 0.989 | 2,036 |
| chunks | ivf | 3 | oversample_then_group | k=3000 | 10 | 0.727 | 8,741 |
| chunks | ivf | 3 | group_by | ef=10 | 10 | 0.990 | 2,398 |

## Reading

- Stock `knn` + `collapse` covers about half the groups on `topics`: its top
  10 hold 5.4 groups on average. Oversampling needs a `k` of 100 to 1,000 per
  10 groups, and the right factor depends on the data.
- With several documents a group, `group_by` is ahead everywhere: on `chunks`
  it reaches 0.86 to 0.99 recall in 0.4 to 2.4 ms, where a 3,000-hit oversample
  stops at 0.73 in 6 to 9 ms (a group's other chunks are rarely in anyone's
  top k). This comes from the two passes: the walks look for each group's best
  document alone, and every document of the best k groups is then scored. The
  single walk bounded by the worst document the groups hold was 2 to 9 times
  slower, with lower recall.
- One document a group: on `topics` `group_by` matches oversampling's
  recall-latency curve and always returns 10 groups; on `chunks`, where small
  groups make a 3x oversample enough, oversampling is up to 1.7x faster at the
  same recall. `group_by` is opt-in, so nothing changes for either.
