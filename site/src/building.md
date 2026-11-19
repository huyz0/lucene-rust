# Build from source

## Tools

| | |
|---|---|
| Rust | the version pinned in `rust-toolchain.toml` (installed automatically by `rustup`) |
| JDK 21+ and Gradle 8.14 | to build the plugin; Gradle uses OpenSearch 3.9.0's bundled JDK 25 as its toolchain |
| Python 3 | the helper scripts and the data fetch step |
| a C toolchain | to link; CMake, Go and Perl as well for the FIPS variant |

## Fetch the data

Dictionaries, ICU packs and other large generated data are not stored in the repository. They are
built from pinned upstream sources, each download verified by SHA-256, by

```sh
scripts/fetch-data.py          # every pack; --list shows them, --pack NAME fetches one
```

The crates' build scripts run it for any file that is missing, so a plain `cargo build` fetches what
it needs. The ICU packs also need a JDK on `PATH`. Downloads are cached in `fixtures/.jars/`.

## Build the plugin

```sh
scripts/lr build              # native library (dist profile) and the plugin zip
```

or step by step:

```sh
scripts/opensearch-dist.sh                 # the pinned OpenSearch jars to compile against
gradle -p opensearch-plugin bundlePlugin   # opensearch-plugin/build/distributions/accelerated-search-<version>.zip
```

`bundlePlugin` builds the native library with the optimized `dist` profile and packages it under
`native/<os>-<arch>/`. `-PcargoProfile=release` builds several times faster for testing;
`-Pfips` builds the FIPS variant. Then [install the zip](get-started.md).

## Run the checks

```sh
scripts/lr doctor             # what this machine has and lacks
scripts/lr test quick         # static checks and the unit tests of the crates you changed
scripts/lr test e2e           # plus one real OpenSearch node (about 20 minutes)
scripts/lr test full          # the whole gate and every node proof
```

Nothing needs Docker; OpenSearch nodes run as plain processes. See
[Testing without Docker](contributing/gates/testing-without-docker.md).

## Build this documentation

```sh
cargo install mdbook --locked
site/build.sh                 # writes site/book/
```
