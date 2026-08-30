# Test/verification environment for lucene-rust.
#
# Exists for one reason: a runaway build or test must not be able to take the
# host down. On WSL2 a process that exhausts the VM's memory gets the whole
# session killed, and recovering it needs a manual `wsl --shutdown` -- so the
# container is given hard memory, CPU, pid and /tmp limits by
# `scripts/docker-test.sh`, and the kernel kills something *inside* the
# container instead.
#
# It also pins the two toolchains the gate is defined against, so
# `gen-fixtures.sh` and `verify-write-path.sh` produce identical bytes here and
# in CI: Rust 1.97.1 (rust-toolchain.toml) and a JDK new enough for Lucene
# 10.5.0, whose jars are baked in so a test run needs no network.

FROM eclipse-temurin:21-jdk

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git pkg-config python3 \
    && rm -rf /var/lib/apt/lists/*

# Pinned to match rust-toolchain.toml. Kept as an ARG so a toolchain bump is a
# one-line diff here and in that file, reviewed together.
ARG RUST_VERSION=1.97.1
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal \
        --default-toolchain "$RUST_VERSION" \
        -c rustfmt -c clippy -c llvm-tools \
    && chmod -R a+w "$RUSTUP_HOME" "$CARGO_HOME"

# Bake the pinned Lucene jars in, so a fixture or verification run is
# reproducible and offline. `lib-lucene-jars.sh` checks $JARS first, so these
# are found without touching Maven Central.
ARG LUCENE_VERSION=10.5.0
ENV LUCENE_JARS=/opt/lucene-jars
RUN mkdir -p "$LUCENE_JARS" && cd "$LUCENE_JARS" \
    && base=https://repo1.maven.org/maven2/org/apache/lucene \
    && for m in lucene-core lucene-analysis-common lucene-queryparser \
                lucene-facet lucene-highlighter lucene-suggest lucene-codecs \
                lucene-backward-codecs lucene-misc lucene-queries; do \
         curl -fsSL -O "$base/$m/$LUCENE_VERSION/$m-$LUCENE_VERSION.jar" || \
           echo "note: $m not published for $LUCENE_VERSION, skipping"; \
       done

# The gate is not just `cargo test`: it includes an aarch64 check-only clippy
# pass (c_char signedness differs by target, so a class of FFI defect is
# invisible on x86_64 alone) and a coverage threshold. Both tools live in the
# image so `scripts/docker-test.sh gate` is the whole gate, not a subset.
RUN rustup target add aarch64-unknown-linux-gnu
RUN cargo install cargo-llvm-cov --locked

# The repo is bind-mounted from the host, so its files are owned by the host
# user while the container runs as root. Git refuses to operate on a repo it
# sees as someone else's, which breaks any script using `git rev-parse` to find
# the repo root (`verify-write-path.sh` does). The mount is explicitly the
# thing we came here to work on, so trusting it is correct.
RUN git config --global --add safe.directory /work

WORKDIR /work
