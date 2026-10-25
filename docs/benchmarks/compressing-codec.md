# CompressingCodec configuration and composition

The pair uses actual Lucene 10.5.0 test codecs and the Rust configuration
adapters. Run `scripts/docker-test.sh bash
scripts/bench-compressing-codec.sh 1000 1500`. Three alternating engine
repetitions use the same two CPUs, Java assertions, the vector incubator module
and a fixed 128 MiB heap.

Twenty-three preconstructed configurations cover all six concrete classes,
custom chunk/docs/shift values, suffixes, and base cases including null modes.
Their complete configuration/identity digest is attached to every mandatory
timed check. Reflection, construction and delegate injection stay outside
timing. Timed cases exercise selected handles and names, all three displays,
nine delegated factories with stable or fresh identities, and one precreated
error whose exact identity must propagate. Full result-byte digests, identity
checks, counters, trace work and result sinks are included on both sides.

The Rust formats are configuration descriptors. Compression factories,
custom-header/geometry stored-field and binary term-vector I/O, default native
codec graphs, SPI and IndexWriter integration remain gaps. Constructors and
random factories are fixture-checked, including failure draw tails, but their
performance is unmeasured: Java constructs a native graph that Rust does not.

## Faithful baseline

Times are median nanoseconds per operation. Cached access performs two SF and
two TV accessors plus a name read; each display creates one complete string;
delegation calls every factory twice; an error operation calls one factory.
Ratios are Java/Rust; noise is same-engine max/min spread.

| Case | Rust ns | Java ns | Ratio | Noise |
| --- | ---: | ---: | ---: | ---: |
| Cached formats/name | 28.819 | 56.758 | 1.97 | 1.01 |
| Stored-fields display | 177.159 | 185.309 | 1.05 | 1.39 |
| Term-vectors display | 176.015 | 167.639 | 0.95 | 1.32 |
| Codec display | 430.304 | 360.656 | 0.84 | 1.15 |
| Delegated factories | 71.525 | 169.333 | 2.37 | 1.14 |
| Reused delegate errors | 26.663 | 35.735 | 1.34 | 2.11 |

The display and error observations are noisy; they do not establish a robust
speed advantage. Cached access and delegation show clear gains in this pair.

## Retained configuration memory

The baseline Rust selected shallow state is 208 bytes, plus 93–132 bytes of
duplicated owned strings and 24 bytes for preset mode instances. Java's
selected shallow state is 112 bytes, static names add no owned payload and a
preset mode adds 16 bytes. Static modes add no per-codec allocation.

The measured duplication motivated sharing one immutable settings allocation
between two distinct, stable format objects, storing custom names once,
deriving standard names from the codec kind and formatting nullable modes
directly. Standard configurations now retain no owned string payload. Custom
input strings remain owned after their caller drops them.

The optimized selected scope is **88 Rust / 112 Java bytes**, or **112 Rust /
128 Java bytes** with a newly allocated preset mode. Both formats remain
distinct objects and share the exact same mode instance. Constructor
validation, random draw order and all Java fixture observations remain equal.

## Optimized measurements

The same pair and settings produce:

| Case | Rust ns | Java ns | Ratio | Noise |
| --- | ---: | ---: | ---: | ---: |
| Cached formats/name | 28.842 | 57.237 | 1.98 | 1.01 |
| Stored-fields display | 152.865 | 137.871 | 0.90 | 1.26 |
| Term-vectors display | 151.339 | 128.605 | 0.85 | 1.41 |
| Codec display | 397.105 | 398.443 | 1.00 | 1.15 |
| Delegated factories | 74.172 | 178.078 | 2.40 | 1.13 |
| Reused delegate errors | 26.671 | 72.222 | 2.71 | 2.03 |

Stored-field and term-vector display medians remain below parity, within their
observed repetition spread. These create fresh strings using Rust's general
formatting machinery versus Java's concatenation; they do not establish a
resolved speed difference. Codec display is likewise within noise. Cached
access and delegation retain clear gains. This pair measures configuration
and composition only; it does not establish compression or I/O performance.

The selected scope counts codec, SF and TV objects, their settings/name storage
and one preset-mode allocation. It excludes native delegate graphs, probe
graphs, allocator metadata, returned components, transient strings, runtime,
peak memory and RSS. Java's referenced static names are shared; Rust-owned
custom input payload must remain counted. This is not complete-codec memory.
