# RandomCodec field routing

Lucene 10.5.0's actual `RandomCodec` is paired with
`lucene-test-framework/src/index/random_codec.rs` by
`bash scripts/bench-random-codec.sh 1000 1500`, run through
`scripts/docker-test.sh`. Three repetitions alternate engine order, pin the
same two CPUs, and use a fixed 128 MiB Java heap, assertions, the vector
incubator module, `tests.nightly=false` and `tests.multiplier=1`.

The inputs are 4,096 precomputed UTF-16 field names, including hash collisions,
supplementary pairs and unpaired surrogates. The cached cases populate their
cache before timing and perform four passes per batch. First-lookup cases
construct a fresh codec before each timed batch, then time one pass including
cache insertion and growth. Constructor, candidate-settings reflection,
identity-table preparation and teardown are excluded. Each measured interval
checks the selected candidate's exact instance ordinal and its precomputed
name/settings token. Every mandatory result check also includes the complete
constructor/settings/input digest, compared across all six engine outputs.

Java's candidate graph includes native formats; Rust's candidates are typed
descriptors. These measurements establish routing behavior only. They do not
measure native postings/doc-values dispatch, suffix grouping, BKD writers,
SPI/IndexWriter integration or complete-codec construction.

## Faithful baseline

Times are median nanoseconds per lookup; ratio is Java/Rust. Noise is the
largest same-engine max/min spread for that case across the three repetitions.

| Case | Rust ns | Java ns | Ratio | Noise |
| --- | ---: | ---: | ---: | ---: |
| Postings, cached | 15.804 | 14.615 | 0.92 | 1.06 |
| Doc values, cached | 17.917 | 14.256 | 0.80 | 1.04 |
| Postings, first lookup | 82.314 | 53.728 | 0.65 | 1.02 |
| Doc values, first lookup | 82.916 | 53.445 | 0.64 | 1.03 |

All four faithful cases are slower than Java beyond the observed repetition
spread.

Java input strings have their cached `hashCode` warmed before timing. Rust
hashes borrowed UTF-16 slices for its native map lookup and computes Java's
hash on misses. Rust's returned `Rc` clone/drop, Java's synchronized map, and
both loops' candidate-identity comparisons and digest arithmetic are included.
Java GC may occur during routing, including reclamation pressure from the
untimed constructor graphs; excluding constructors does not remove that cost.

## Optimized measurements and remaining gap

The retained change uses boxed keys with the standard map and hasher. The
same full pair produces:

| Case | Rust ns | Java ns | Ratio | Noise |
| --- | ---: | ---: | ---: | ---: |
| Postings, cached | 16.623 | 14.425 | 0.87 | 1.06 |
| Doc values, cached | 18.337 | 14.234 | 0.78 | 1.03 |
| Postings, first lookup | 78.145 | 54.030 | 0.69 | 1.03 |
| Doc values, first lookup | 78.854 | 53.749 | 0.68 | 1.03 |

Cached times differ from the faithful baseline within the observed spreads;
first lookups improve modestly. All four cases remain below Java beyond their
current repetition spread. This is a documented performance gap.

The API takes borrowed UTF-16 slices. Every native lookup hashes their content;
a miss also computes the Java routing hash, copies the key to retain it after
the borrow ends, and hashes it again on insertion. Java retains the exact
external strings and reuses their warmed cached hash codes. A separate probe
on the same keys measured Java-compatible content hashing alone at
2.64–2.74 ns/key, standard-map cached lookup at 10.29–10.98 ns/key, and owned
key copying at 6.9–7.2 ns/key. These isolated measurements are not additive
attribution of the complete benchmark gap; map growth and reference handling
are also included in the pair.

Several native hash variants and cross-crate inlining were measured and
rejected: one reduced first lookups to about 70 ns while worsening cached
lookups to about 19 ns, and removing its finalizer with a different short-tail
path worsened both paths. The standard map keeps the simpler implementation
and the memory reduction. Matching Java's string ownership and hash reuse
would require a separately measured owned/shared field-key API; it cannot
silently alias or retain an arbitrary caller's borrowed slice. That wider API
change is left open rather than adding another field representation to this
descriptor-only codec. No native format throughput or construction advantage
is claimed.

## Retained key storage

The shared external inputs contain 21,136 logical UTF-16 units. Each populated
Rust cache additionally owns 42,272 requested bytes of copied UTF-16 text and
65,536 bytes of inline `Box<[u16]>` key records, reduced from the faithful
baseline's 98,304 bytes of `Vec<u16>` records. Java's map retains the exact
external `String` instances, checked by identity, so it adds zero copied text
or Rust-style boxed-slice records.

These rows are scoped payload/key-record counts, not total cache memory. They
exclude table/bucket/node/reference storage, capacity slack and padding,
candidate graphs, physical Java string encoding, allocator/runtime overhead,
peak memory and RSS. The slice-taking Rust API copies a key to retain it after
the caller's borrow ends; this difference must remain visible in any memory
comparison.
