# AssertingCodec composition

The pair is Lucene 10.5.0's actual `AssertingCodec` and the explicit-provider
Rust composition. Run `scripts/docker-test.sh bash
scripts/bench-asserting-codec.sh 1000 1500`. Three repetitions alternate engine
order on the same two CPUs. Java uses assertions, the vector incubator module,
a fixed 128 MiB heap, nightly false and multiplier one.

Eleven accessors are exercised twice per step under suppression masks 0, 21 and
63. Selector cases cover three kinds, four fields including null, and direct,
internal-router and public paths. Other cases render live delegate displays
or reuse one precreated delegate error. Every result check covers exact cached
or delegated identity, ordered factory/hook traces and callback counts. Seven
checks are mandatory in every engine repetition. Identity checks, digest work,
Java volatile sinks and Rust `black_box` are included in the timings.

The optimized observer checks and consumes the concrete returned handle
directly. Format identity compares allocation data addresses against retained
expected handles; router identity uses `same_instance`. This avoids a timed
benchmark-only tagged union without removing a getter, returned-handle sink,
identity check, counter or digest event. Complete per-case result digests
must match Java in all six outputs. Consequently, speed changes from
the faithful baseline include observer overhead changes and cannot be
attributed solely to the library representation.

Providers, delegates and expected handles are prepared outside timing. Java's
defaults have their real native inner formats; Rust constructs the eight
existing asserting adapters around opaque providers. Delegate points return a
stable prebuilt handle on both sides. These are composition measurements;
native graph construction, fresh native points allocation, component I/O,
SPI and IndexWriter integration are outside the comparison.

## Faithful baseline

Times are median nanoseconds per complete step. A component step calls all
eleven accessors twice; a selector step performs all 36 selections; a display
step creates three strings; an error step calls all eleven accessors under
all three masks. Ratios are Java/Rust; noise is same-engine max/min spread.

| Case | Rust ns | Java ns | Ratio | Noise |
| --- | ---: | ---: | ---: | ---: |
| Cached accessors | 58.403 | 116.316 | 1.99 | 1.01 |
| Suppressed accessors | 68.769 | 143.453 | 2.09 | 1.02 |
| Mixed suppression | 63.431 | 141.931 | 2.24 | 1.01 |
| Selector hooks | 154.651 | 323.060 | 2.09 | 1.06 |
| Public KNN bypass | 146.508 | 341.561 | 2.33 | 1.15 |
| Display | 283.004 | 771.357 | 2.73 | 1.02 |
| Reused delegate errors | 94.319 | 852.576 | 9.04 | 1.08 |

## Optimized measurements

The same complete workloads, with typed router handles and the concrete
observer described above, produce:

| Case | Rust ns | Java ns | Ratio | Noise |
| --- | ---: | ---: | ---: | ---: |
| Cached accessors | 36.245 | 117.177 | 3.23 | 1.02 |
| Suppressed accessors | 45.960 | 136.255 | 2.96 | 1.10 |
| Mixed suppression | 41.538 | 128.285 | 3.09 | 1.13 |
| Selector hooks | 115.515 | 326.003 | 2.82 | 1.09 |
| Public KNN bypass | 107.985 | 342.879 | 3.18 | 1.02 |
| Display | 288.201 | 759.117 | 2.63 | 1.04 |
| Reused delegate errors | 69.950 | 858.311 | 12.27 | 1.50 |

All cases exceed parity beyond the observed repetition spread. Reused-error
timings remain comparatively noisy. Constructor performance is unmeasured
because the native graph scopes differ.

## Retained adapter memory

The baseline selected scope is 384 Rust bytes versus 328 Java bytes. Rust
retains 48 bytes of codec state, 184 bytes in eight asserting wrapper nodes,
72 bytes in three router nodes and 80 bytes of shared selector state. Java
retains 72 bytes of codec state, 152 bytes in eight wrapper nodes, 72 bytes in
three routers and 32 bytes in its suppression EnumSet.

The measured excess motivated replacing separately allocated Rust routers
with inline typed handles sharing selector state. The optimized selected
scope is **312 Rust / 328 Java bytes**: the 48-byte codec, 184 bytes in eight
wrapper nodes and 80 bytes of shared selector state. There are no separately
allocated router nodes. Each typed router's logical identity is its router
type plus the shared selector-state instance; clones retain that identity,
including after the codec is dropped. Selected format `Rc` identity remains
exact. The Java object breakdown is unchanged.

This scope uses a plain default-selector codec. It counts each retained node
once, including Java object headers and Rust reference counters/alignment.
Native default/delegate inner graphs, instrumentation, alternate hook formats,
static names, transient strings, allocator metadata, runtime/GC, peak memory
and RSS are excluded. It is not a full codec-memory comparison.
